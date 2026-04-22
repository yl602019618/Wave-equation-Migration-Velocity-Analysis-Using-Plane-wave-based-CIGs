"""PWEMVA outer loop (torch GPU) — port of ``pwmva_python.inversion``.

Differences from the numpy version:

  - No ProcessPoolExecutor. All per-plane-wave work is done in GPU batches
    of size ``batch`` (default 8); all 41 PWs share one velocity so the PML
    coefficients need to be built only once per RTM / wavepath call.
  - Dynamic warping stays on CPU (numpy + numba) — it is already fast
    (~1.5 s / iter) and bit-perfect aligned to Fortran.
  - ``theta_frozen`` semantics match the Fortran / patched numpy: ``theta`` is
    computed once at iter begin from ``v_k`` and reused across all line-search
    trials; ``slowmig`` is recomputed per trial from ``v_trial`` (Fortran line 475).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch

from . import io, image_ops as ops
from .device import pick_device, DTYPE
from .source import ricker
from .fd2d import pw1_rtm_batch, pw1_wavepath_batch

# reuse numpy warping (numba, already bit-perfect)
from pwmva.warping import warping_misfit
from pwmva.config import Parfile
from pwmva.geom   import PW1Coord


PILOT_0B = 20     # Fortran is_pilot=21 (1-based) → 20 here
LAGZ     = 17
EPS_MASK = 30
TOP_MUTE = 30
N_STRAIN = 6


# ===================================================================
# Helpers to apply mask / smooth pipelines across the (NS, NZ, NX) volume
# ===================================================================

def _apply_image_masks(image_tot: torch.Tensor, theta: np.ndarray,
                       p_arr: np.ndarray, slowmig: torch.Tensor) -> torch.Tensor:
    """image_tot: (NS, NZ, NX). In-place-style on a clone."""
    out = image_tot.clone()
    ns = out.shape[0]
    for i in range(ns):
        out[i] = ops.maskimage(out[i], float(theta[i]), EPS_MASK)
    for i in range(ns):
        out[i] = ops.snell_mute(out[i], float(p_arr[i]), slowmig)
    for i in range(ns):
        out[i] = ops.top_mute(out[i], TOP_MUTE)
    return out


def _apply_shift_masks(shift: torch.Tensor, theta: np.ndarray,
                       p_arr: np.ndarray, slowmig: torch.Tensor) -> torch.Tensor:
    out = shift.clone()
    ns = out.shape[0]
    for i in range(ns):
        out[i] = ops.top_mute(out[i], TOP_MUTE)
    for i in range(ns):
        out[i] = ops.maskimage(out[i], float(theta[i]), EPS_MASK)
    for i in range(ns):
        out[i] = ops.snell_mute(out[i], float(p_arr[i]), slowmig)
    return out


# ===================================================================
# Geometry + data holders — passed through run_iteration
# ===================================================================

@dataclass
class InvContext:
    """All per-run invariants kept on GPU / CPU once."""
    par: Parfile
    coord: PW1Coord
    device: torch.device
    # GPU tensors
    src: torch.Tensor         # (nw,)
    p_ray: torch.Tensor       # (NS,)
    xref_1: torch.Tensor      # (NS,)
    zp1: torch.Tensor         # (NS,)
    xg: torch.Tensor          # (NS, ng)
    zg: torch.Tensor          # (NS, ng)
    seis: torch.Tensor        # (NS, nt, ng) observed data
    # config
    batch: int = 8


def build_context(par: Parfile, coord: PW1Coord,
                  cpgs: list[np.ndarray] | np.ndarray,
                  src_np: np.ndarray,
                  device: torch.device | str | None = None,
                  batch: int = 8) -> InvContext:
    device = pick_device(str(device)) if device is None or isinstance(device, str) else device
    ns = coord.npro
    ng = int(coord.ng[0])
    p_ray  = torch.from_numpy(coord.p.astype(np.float32)).to(device)
    xref_1 = torch.from_numpy(coord.xref_1.astype(np.float32)).to(device)
    zp1    = torch.from_numpy(coord.zp1.astype(np.float32)).to(device)
    xg = torch.from_numpy(coord.xg[:ng, :].T.astype(np.float32)).to(device)
    zg = torch.from_numpy(coord.zg[:ng, :].T.astype(np.float32)).to(device)
    if isinstance(cpgs, list):
        cpgs_arr = np.stack(cpgs, axis=0)
    else:
        cpgs_arr = cpgs
    seis = torch.from_numpy(cpgs_arr.astype(np.float32)).to(device)    # (NS, nt, ng)
    src = torch.from_numpy(np.asarray(src_np, dtype=np.float32)).to(device)
    return InvContext(par=par, coord=coord, device=device,
                      src=src, p_ray=p_ray, xref_1=xref_1, zp1=zp1,
                      xg=xg, zg=zg, seis=seis, batch=batch)


# ===================================================================
# Step 1: image_tot via batched RTM
# ===================================================================

@torch.no_grad()
def compute_image_tot(v: torch.Tensor, ctx: InvContext) -> torch.Tensor:
    """Returns image_tot (NS, NZ, NX) = -∂_z RTM(v, cpg_is) per PW."""
    par = ctx.par
    ns = ctx.p_ray.shape[0]
    nz, nx = v.shape
    image_tot = torch.zeros((ns, nz, nx), dtype=DTYPE, device=ctx.device)
    for b0 in range(0, ns, ctx.batch):
        b1 = min(ns, b0 + ctx.batch)
        res = pw1_rtm_batch(
            v, par.dx, par.dt, par.nt, par.npml, ctx.src,
            ctx.p_ray[b0:b1], ctx.xref_1[b0:b1], ctx.zp1[b0:b1],
            ctx.xg[b0:b1], ctx.zg[b0:b1], ctx.seis[b0:b1],
            xs_taper_grid=40, dt_record=6,
        )
        image_tot[b0:b1] = -ops.z_derivative_2d(res.img, fd_order=4)
    return image_tot


# ===================================================================
# Step 3: warping (CPU numba) + misfit
# ===================================================================

def compute_warp_misfit(image_masked: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Delegates to numpy warping (1.5 s for 41 PWs, fast enough)."""
    image_np = image_masked.detach().to("cpu").numpy().transpose(1, 2, 0)   # (nz, nx, ns)
    shift_np, misfit = warping_misfit(image_np, is_pilot_0b=PILOT_0B,
                                      lagz=LAGZ, n_strain=N_STRAIN)
    shift_t = torch.from_numpy(shift_np.transpose(2, 0, 1)).to(image_masked.device)
    return shift_t, float(misfit)


# ===================================================================
# Step 5–9: gradient
# ===================================================================

@torch.no_grad()
def compute_gradient(v: torch.Tensor, image_masked: torch.Tensor,
                     shift_masked: torch.Tensor, ctx: InvContext) -> torch.Tensor:
    par = ctx.par
    ns, nz, nx = image_masked.shape

    # num[is] before any smoothing of shift
    num = torch.tensor([ops.nzmean2d(shift_masked[i]) for i in range(ns)],
                       dtype=DTYPE, device=ctx.device)

    # smooth the shift per PW
    shift_sm = torch.empty_like(shift_masked)
    for i in range(ns):
        shift_sm[i] = ops.gaussian_smooth_2d(5.0, 20.0, shift_masked[i])

    # AGC weighting: refl = shift_sm * imageagc(image)
    refl_all = torch.zeros_like(shift_sm)
    for i in range(ns):
        if i == PILOT_0B:
            continue
        agc = ops.imageagc(image_masked[i], np_pad=30)
        refl_all[i] = shift_sm[i] * agc

    # background v for wavepath
    v_smooth = ops.vel_smooth(v, nzs=11, nxs=21, niter=3)

    # wavepath over non-pilot PWs, batched
    non_pilot = [i for i in range(ns) if i != PILOT_0B]
    gk1_tot = torch.zeros((nz, nx), dtype=DTYPE, device=ctx.device)
    for b0 in range(0, len(non_pilot), ctx.batch):
        sub = non_pilot[b0:b0 + ctx.batch]
        sub_idx = torch.tensor(sub, device=ctx.device)
        res = pw1_wavepath_batch(
            v=v_smooth,
            refl=refl_all[sub_idx],
            dx=par.dx, dt=par.dt, nt=par.nt, npml=par.npml, src=ctx.src,
            p_ray=ctx.p_ray[sub_idx], xref_1=ctx.xref_1[sub_idx],
            zp1_phys=ctx.zp1[sub_idx],
            xg=ctx.xg[sub_idx], zg=ctx.zg[sub_idx],
            seis=ctx.seis[sub_idx],
            xs_taper_grid=40, dt_record=6,
        )
        # res.img is already per-PW-normalised by max|img|.
        # Accumulate Σ res.img[k] * num[i]
        for k, i in enumerate(sub):
            gk1_tot = gk1_tot + res.img[k] * num[i]

    # post-process: top mute + gaussian smooth
    dk1_tot = gk1_tot.clone()
    dk1_tot[:20, :] = 0.0
    dk1_tot = ops.gaussian_smooth_2d(2.0, 20.0, dk1_tot)
    return dk1_tot


# ===================================================================
# Velocity update + line search
# ===================================================================

def velocity_update(v: torch.Tensor, grad: torch.Tensor,
                    alpha_tot: float, fff: float,
                    vmin: float, vmax: float) -> torch.Tensor:
    s = 1.0 / v.to(DTYPE)
    s1 = s - float(alpha_tot) * float(fff) * grad.to(DTYPE)
    v_new = 1.0 / s1
    return torch.clamp(v_new, min=float(vmin), max=float(vmax)).to(DTYPE)


@torch.no_grad()
def trial_misfit(v_trial: torch.Tensor, theta_frozen: np.ndarray,
                 ctx: InvContext) -> float:
    """Recompute steps 1-3 with v_trial.

    Theta is FROZEN (computed once at iter begin from v_{k-1}) — matches Fortran.
    slowmig is recomputed per trial from v_trial.
    """
    image_tot = compute_image_tot(v_trial, ctx)
    slowmig = (1.0 / v_trial).to(DTYPE)
    image_masked = _apply_image_masks(image_tot, theta_frozen, ctx.coord.p, slowmig)
    shift, _ = compute_warp_misfit(image_masked)
    shift_masked = _apply_shift_masks(shift, theta_frozen, ctx.coord.p, slowmig)
    return float((shift_masked ** 2).sum().item())


@torch.no_grad()
def line_search(v: torch.Tensor, grad: torch.Tensor, misfit_tot: float,
                theta_frozen: np.ndarray, ctx: InvContext,
                vmin: float, vmax: float, log=print) -> tuple[float, float]:
    s = 1.0 / v.to(DTYPE)
    alpha_tot = float(s.max().item() / ops.mean_abs(grad))

    f1 = 0.01
    v1 = velocity_update(v, grad, alpha_tot, f1, vmin, vmax)
    misfit1 = trial_misfit(v1, theta_frozen, ctx)
    log(f"  line-search: f1={f1:.3e}  misfit1={misfit1:.3e}  (misfit_tot={misfit_tot:.3e})")

    if misfit1 > misfit_tot:
        while misfit1 > misfit_tot and f1 > 1e-7:
            f1 *= 0.5
            v1 = velocity_update(v, grad, alpha_tot, f1, vmin, vmax)
            misfit1 = trial_misfit(v1, theta_frozen, ctx)
            log(f"    backtrack: f1={f1:.3e}  misfit1={misfit1:.3e}")
        return f1, alpha_tot
    else:
        f2 = 2.0 * f1
        v2 = velocity_update(v, grad, alpha_tot, f2, vmin, vmax)
        misfit2 = trial_misfit(v2, theta_frozen, ctx)
        log(f"  line-search: f2={f2:.3e}  misfit2={misfit2:.3e}")
        fff = f1 if misfit2 > misfit1 else f2
        return fff, alpha_tot


# ===================================================================
# Outer loop
# ===================================================================

@dataclass
class IterState:
    v: torch.Tensor          # (NZ, NX) GPU tensor
    iter_idx: int            # 1-based
    misfit: float = 0.0
    fff: float = 0.0
    grad: torch.Tensor | None = None


@torch.no_grad()
def run_iteration(state: IterState, ctx: InvContext,
                  debug_dir: Path | None = None, log=print,
                  skip_3d_dump: bool = False) -> IterState:
    v = state.v
    iii = state.iter_idx
    par = ctx.par
    log(f"\n=== Iteration {iii} ===  device={ctx.device}  batch={ctx.batch}")

    # 1-2: image_tot + masks
    t0 = time.time()
    image_tot = compute_image_tot(v, ctx)
    # theta is FROZEN for this iteration (Fortran semantics)
    theta = np.arcsin(ctx.coord.p * float(v[0, :].mean().item())).astype(np.float32)
    slowmig = (1.0 / v).to(DTYPE)
    image_masked = _apply_image_masks(image_tot, theta, ctx.coord.p, slowmig)
    if debug_dir is not None and not skip_3d_dump:
        # write as (NZ, NX, NS) to match Fortran ordering
        io.write3d(debug_dir / f"image_tot_{iii}.bin", image_tot.permute(1, 2, 0).contiguous())
    log(f"  image_tot + mask: {time.time()-t0:.1f} s")

    # 3-4: warping + masks
    t1 = time.time()
    shift, _ = compute_warp_misfit(image_masked)
    shift_masked = _apply_shift_masks(shift, theta, ctx.coord.p, slowmig)
    misfit_tot = float((shift_masked ** 2).sum().item())
    log(f"  warping: {time.time()-t1:.1f} s   misfit_tot = {misfit_tot:.3e}")
    if debug_dir is not None and not skip_3d_dump:
        io.write3d(debug_dir / f"shift_{iii}.bin", shift_masked.permute(1, 2, 0).contiguous())

    # 5-9: gradient
    t2 = time.time()
    grad = compute_gradient(v, image_masked, shift_masked, ctx)
    log(f"  gradient: {time.time()-t2:.1f} s")
    if debug_dir is not None:
        io.write2d(debug_dir / f"gradient_{iii}.bin", grad)

    # free image-domain tensors we no longer need before line search
    del image_tot, image_masked, shift, shift_masked
    torch.cuda.empty_cache()

    # 10-11: line search + update — pass the FROZEN theta
    t3 = time.time()
    fff, alpha_tot = line_search(v, grad, misfit_tot, theta, ctx,
                                 par.vmin, par.vmax, log)
    log(f"  line search: {time.time()-t3:.1f} s   fff={fff:.3e}  alpha_tot={alpha_tot:.3e}")
    v_new = velocity_update(v, grad, alpha_tot, fff, par.vmin, par.vmax)

    # 12: smooth every 3 iters at iter mod 3 == 1
    if iii % 3 == 1:
        v_new = ops.vel_smooth(v_new, nzs=1, nxs=51, niter=3)
    if debug_dir is not None:
        io.write2d(debug_dir / f"velinv_{iii}.bin", v_new)
    return IterState(v=v_new, iter_idx=iii + 1, misfit=misfit_tot,
                     fff=fff, grad=grad)
