"""Batched WEMVA wavepath gradient kernel.

Port of ``pwmva_python.fd2d.wavepath.pw1_wavepath``. For B plane waves
simultaneously we compute:

  pp   : source-side background
  ppb  : source-side Born-scattered field, source = (∇²pp)·refl·β_dt
  pq   : receiver-side background, back-prop of observed data
  imgr = Σ_t pq · ppb                       (per PW, normalised by |·|_max)

The receiver-side Born (``pqb``) contribution is dropped — Fortran's final
assignment uses only ``imgr``.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from ..device import DTYPE
from .pml import build_alpha_temp, build_taper
from .stencil import kernel_step, pertubation_ic2


@dataclass
class WavepathBatchResult:
    img: torch.Tensor     # (B, nz, nx) — each PW normalised by its own max|img|


@torch.no_grad()
def pw1_wavepath_batch(
    v: torch.Tensor,             # (nz, nx)  smoothed velocity for background
    refl: torch.Tensor,          # (B, nz, nx)  virtual reflectivity per PW
    dx: float, dt: float, nt: int, npml: int,
    src: torch.Tensor,           # (nw,)
    p_ray:    torch.Tensor,      # (B,)
    xref_1:   torch.Tensor,      # (B,)
    zp1_phys: torch.Tensor,      # (B,)
    xg: torch.Tensor,            # (B, ng)
    zg: torch.Tensor,            # (B, ng)
    seis: torch.Tensor,          # (B, nt, ng)
    xs_taper_grid: int = 40,
    dt_record: int = 6,
) -> WavepathBatchResult:
    device = v.device
    B = p_ray.shape[0]
    nz, nx = v.shape
    nz_pml = nz + 2 * npml
    nx_pml = nx + 2 * npml

    alpha, temp1, temp2, beta_dt = build_alpha_temp(v, dx, dt, npml)
    taper = build_taper(nx, xs_taper_grid, device=device)

    # pad refl per-PW to PML grid (replicate edges)
    refl_pml = F.pad(refl.unsqueeze(1), (npml, npml, npml, npml),
                     mode="replicate").squeeze(1)               # (B, nz_pml, nx_pml)

    zp1_scalar = int(torch.round(zp1_phys[0] / dx).item()) + npml
    igx = (npml + (xg / dx).to(torch.int64))
    igz = (npml + (zg / dx).to(torch.int64))
    ng = xg.shape[1]

    i_phys = torch.arange(nx, dtype=torch.float64, device=device)
    delays = torch.round(p_ray.double().unsqueeze(1)
                         * (i_phys.unsqueeze(0) * dx - xref_1.double().unsqueeze(1))
                         / dt).to(torch.int64)

    ix_pml = torch.arange(npml, npml + nx, dtype=torch.int64, device=device)
    bd_row = beta_dt[zp1_scalar, ix_pml]
    bd_recv = beta_dt[igz, igx]                                 # (B, ng)

    src = src.to(device=device, dtype=DTYPE)
    seis = seis.to(device=device, dtype=DTYPE)
    nw = src.shape[0]

    iz0, iz1 = 4, nz_pml - 4
    ix0, ix1 = 4, nx_pml - 4

    nt_record = int(round(nt / dt_record)) + 1
    ppb_store = torch.zeros((B, nz, nx, nt_record), dtype=DTYPE, device=device)
    pp_store  = torch.zeros((B, nz, nx, nt_record), dtype=DTYPE, device=device)

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, ng)
    b_src     = torch.arange(B, device=device).unsqueeze(1).expand(-1, nx)
    ix_src    = ix_pml.unsqueeze(0).expand(B, -1)

    # ---------- 1. Forward: background + Born ----------
    p0  = torch.zeros((B, nz_pml, nx_pml), dtype=DTYPE, device=device)
    p1  = torch.zeros_like(p0)
    p   = torch.zeros_like(p0)
    pb0 = torch.zeros_like(p0)
    pb1 = torch.zeros_like(p0)
    pb  = torch.zeros_like(p0)

    for it in range(1, nt + 1):
        # background
        kernel_step(p, p0, p1, alpha, temp1, temp2, iz0, iz1, ix0, ix1)
        itt = it - delays
        valid = (itt >= 1) & (itt <= nw)
        idx_t = torch.clamp(itt - 1, 0, nw - 1)
        src_val = torch.where(valid, src[idx_t], torch.zeros_like(src[idx_t]))
        add = (bd_row.unsqueeze(0) * src_val * taper.unsqueeze(0))
        p[b_src, zp1_scalar, ix_src] = p[b_src, zp1_scalar, ix_src] + add

        # Born scattered: NOTE this uses p1 (before rotation) = previous-timestep current-time field
        kernel_step(pb, pb0, pb1, alpha, temp1, temp2, iz0, iz1, ix0, ix1)
        pertubation_ic2(p1, refl_pml, beta_dt, pb)

        it_rec = int(round(it / dt_record))
        ppb_store[:, :, :, it_rec] = pb[:, npml:npml + nz, npml:npml + nx]
        pp_store [:, :, :, it_rec] = p [:, npml:npml + nz, npml:npml + nx]

        p0,  p1,  p  = p1,  p,  p0
        pb0, pb1, pb = pb1, pb, pb0

    # Free pp_store early (we only need ppb_store for the final correlation)
    del pp_store, pb, pb0, pb1
    torch.cuda.empty_cache()

    # ---------- 2. Back: pq (background only) ----------
    p0.zero_(); p1.zero_(); p.zero_()
    imgr = torch.zeros((B, nz, nx), dtype=DTYPE, device=device)

    for it in range(nt, 0, -1):
        kernel_step(p, p0, p1, alpha, temp1, temp2, iz0, iz1, ix0, ix1)
        add = bd_recv * seis[:, it - 1, :]
        p[batch_idx, igz, igx] = p[batch_idx, igz, igx] + add

        it_rec = int(round(it / dt_record))
        imgr = imgr + p[:, npml:npml + nz, npml:npml + nx] * ppb_store[:, :, :, it_rec]

        p0, p1, p = p1, p, p0

    # ---------- 3. Normalise per-PW ----------
    m = imgr.abs().view(B, -1).max(dim=1).values.clamp(min=1e-30)      # (B,)
    imgr = imgr / m.view(B, 1, 1)
    return WavepathBatchResult(img=imgr)
