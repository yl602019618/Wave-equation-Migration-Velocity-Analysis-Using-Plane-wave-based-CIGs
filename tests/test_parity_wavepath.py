"""T5 parity: batched wavepath kernel.

Two-level comparison:

(a) Single-PW wavepath call vs numpy ``pwmva_python.fd2d.pw1_wavepath`` — tight
    correlation ≥ 0.9999.

(b) End-to-end gradient_1.bin: full RTM → mask → warping → shift_sm → AGC →
    wavepath → sum-over-PWs → top-mute → gauss-smooth pipeline. Reuses the
    numpy warping module (already bit-perfect aligned to Fortran).
"""
from pathlib import Path
import time
import numpy as np
import torch

from pwmva_torch.device import pick_device
from pwmva_torch.source import ricker
from pwmva_torch.fd2d.pw1_rtm import pw1_rtm_batch
from pwmva_torch.fd2d.wavepath import pw1_wavepath_batch
from pwmva_torch import image_ops as ops_t

# numpy oracles
from pwmva.io import read2d as read2d_np, read3d as read3d_np
from pwmva.config import read_parfile
from pwmva.geom  import read_pw1_coord
from pwmva.fd2d.wavepath import pw1_wavepath as pw1_wavepath_np
from pwmva import image_ops as ops_np
from pwmva.source import ricker as ricker_np
from pwmva.warping import warping_misfit

from _viz import plot_2d_compare, pearson, rms_rel

DEVICE = pick_device()
NZ, NX, NS = 201, 801, 41
PILOT_0B, LAGZ, EPS_MASK, TOP_MUTE, N_STRAIN = 20, 17, 30, 30, 6


def _gather_coord_tensors(coord, device, idxs=None):
    if idxs is None:
        idxs = list(range(coord.npro))
    ng = int(coord.ng[idxs[0]])
    p_ray = torch.tensor([float(coord.p[i]) for i in idxs], dtype=torch.float32, device=device)
    xref  = torch.tensor([float(coord.xref_1[i]) for i in idxs], dtype=torch.float32, device=device)
    zp1   = torch.tensor([float(coord.zp1[i]) for i in idxs], dtype=torch.float32, device=device)
    xg = torch.tensor(np.stack([coord.xg[:ng, i] for i in idxs]), dtype=torch.float32, device=device)
    zg = torch.tensor(np.stack([coord.zg[:ng, i] for i in idxs]), dtype=torch.float32, device=device)
    return p_ray, xref, zp1, xg, zg, ng


def test_single_pw_wavepath(model_dir, csg_dir, rerun_dir, working_dir, viz_dir):
    """One PW through the full wavepath with refl = 0.01 * constant reflectivity —
    cheap sanity check that the batched kernel matches numpy."""
    par = read_parfile(working_dir / "parfile_pwmva_warp_rerun.sh")
    coord = read_pw1_coord(model_dir / "coord_pw_45t45.dat")
    velh = read2d_np(model_dir / "velh_201x801x2.5m.bin", NZ, NX).astype(np.float32)
    v_smooth = ops_np.vel_smooth(velh, nzs=11, nxs=21, niter=3)

    # synthetic refl
    refl_np = np.zeros((NZ, NX), dtype=np.float32)
    refl_np[80:90, 200:600] = 0.05

    i = 10  # off-pilot PW
    ng = int(coord.ng[i])
    cpg = read2d_np(csg_dir / "cpg" / f"cpg_{i+1}.bin", par.nt, ng).astype(np.float32)

    res_np = pw1_wavepath_np(
        v=v_smooth, refl=refl_np,
        dx=par.dx, dt=par.dt, nt=par.nt, npml=par.npml,
        src=ricker_np(par.nw, par.dt, par.freq),
        p_ray=float(coord.p[i]), xref_1=float(coord.xref_1[i]),
        zp1_phys=float(coord.zp1[i]),
        xg=coord.xg[:ng, i], zg=coord.zg[:ng, i],
        seis=cpg,
    ).img

    p_ray, xref, zp1, xg, zg, _ = _gather_coord_tensors(coord, DEVICE, idxs=[i])
    v_s_t = torch.from_numpy(v_smooth).to(DEVICE)
    refl_t = torch.from_numpy(refl_np).unsqueeze(0).to(DEVICE)
    seis_t = torch.from_numpy(cpg).unsqueeze(0).to(DEVICE)
    src_t = ricker(par.nw, par.dt, par.freq, device=DEVICE)

    res_t = pw1_wavepath_batch(
        v=v_s_t, refl=refl_t,
        dx=par.dx, dt=par.dt, nt=par.nt, npml=par.npml, src=src_t,
        p_ray=p_ray, xref_1=xref, zp1_phys=zp1, xg=xg, zg=zg, seis=seis_t,
    )
    res_t_np = res_t.img[0].detach().to("cpu").numpy()

    p = pearson(res_np, res_t_np); r = rms_rel(res_np, res_t_np)
    print(f"\n[T5/single-PW] pearson={p:.6f}  rms_rel={r:.3e}  "
          f"max np={np.abs(res_np).max():.3e}  max torch={np.abs(res_t_np).max():.3e}")
    assert p > 0.9999
    plot_2d_compare(res_np, res_t_np,
                    [f"np wavepath is={i+1}", "torch wavepath", "diff"],
                    viz_dir / f"t5_single_pw_is{i+1}.png",
                    clip_percentile=99.5)


def _apply_image_masks_t(image_tot_t, theta, p_arr, slowmig):
    """image_tot_t: (NS, NZ, NX) torch — maskimage → snell_mute → top_mute."""
    out = image_tot_t.clone()
    for i in range(out.shape[0]):
        out[i] = ops_t.maskimage(out[i], float(theta[i]), EPS_MASK)
    for i in range(out.shape[0]):
        out[i] = ops_t.snell_mute(out[i], float(p_arr[i]), slowmig)
    for i in range(out.shape[0]):
        out[i] = ops_t.top_mute(out[i], TOP_MUTE)
    return out


def _apply_shift_masks_t(shift_t, theta, p_arr, slowmig):
    out = shift_t.clone()
    for i in range(out.shape[0]):
        out[i] = ops_t.top_mute(out[i], TOP_MUTE)
    for i in range(out.shape[0]):
        out[i] = ops_t.maskimage(out[i], float(theta[i]), EPS_MASK)
    for i in range(out.shape[0]):
        out[i] = ops_t.snell_mute(out[i], float(p_arr[i]), slowmig)
    return out


def test_endtoend_gradient_1(model_dir, csg_dir, working_dir, rerun_dir, viz_dir):
    """End-to-end: reproduce gradient_1.bin from velh + cpgs on GPU."""
    par = read_parfile(working_dir / "parfile_pwmva_warp_rerun.sh")
    coord = read_pw1_coord(model_dir / "coord_pw_45t45.dat")
    v_np = read2d_np(model_dir / "velh_201x801x2.5m.bin", NZ, NX).astype(np.float32)
    ng = int(coord.ng[0])

    cpgs_np = np.stack([
        read2d_np(csg_dir / "cpg" / f"cpg_{i+1}.bin", par.nt, ng).astype(np.float32)
        for i in range(NS)
    ])

    device = DEVICE
    v_t = torch.from_numpy(v_np).to(device)
    src = ricker(par.nw, par.dt, par.freq, device=device)
    seis_all = torch.from_numpy(cpgs_np.astype(np.float32)).to(device)
    p_ray, xref, zp1, xg, zg, _ = _gather_coord_tensors(coord, device)

    # ---------- Step 1: image_tot via batched RTM ----------
    t0 = time.time()
    image_tot = torch.zeros((NS, NZ, NX), dtype=torch.float32, device=device)
    B = 8
    for b0 in range(0, NS, B):
        b1 = min(NS, b0 + B)
        res = pw1_rtm_batch(v_t, par.dx, par.dt, par.nt, par.npml, src,
                            p_ray[b0:b1], xref[b0:b1], zp1[b0:b1],
                            xg[b0:b1], zg[b0:b1], seis_all[b0:b1],
                            xs_taper_grid=40, dt_record=6)
        image_tot[b0:b1] = -ops_t.z_derivative_2d(res.img, fd_order=4)
    torch.cuda.synchronize()
    t_rtm = time.time() - t0
    print(f"\n[T5/e2e] RTM 41 PW: {t_rtm:.2f} s")

    # ---------- Step 2: image masks ----------
    theta = np.arcsin(coord.p * v_np[0, :].mean()).astype(np.float32)
    slowmig_t = (1.0 / v_t).to(torch.float32)
    image_masked = _apply_image_masks_t(image_tot, theta, coord.p, slowmig_t)

    # ---------- Step 3: warping (CPU numba, reuse) ----------
    image_masked_np = image_masked.detach().to("cpu").numpy().transpose(1, 2, 0)  # (nz, nx, ns)
    shift_np, _ = warping_misfit(image_masked_np, is_pilot_0b=PILOT_0B,
                                 lagz=LAGZ, n_strain=N_STRAIN)
    shift_t = torch.from_numpy(shift_np.transpose(2, 0, 1)).to(device)

    # ---------- Step 4: shift masks; num[is] ----------
    shift_masked = _apply_shift_masks_t(shift_t, theta, coord.p, slowmig_t)
    num = torch.tensor([ops_t.nzmean2d(shift_masked[i]) for i in range(NS)],
                       dtype=torch.float32, device=device)

    # ---------- Step 5: smooth shift + AGC weighting per PW ----------
    shift_sm = torch.empty_like(shift_masked)
    for i in range(NS):
        shift_sm[i] = ops_t.gaussian_smooth_2d(5.0, 20.0, shift_masked[i])
    refl_all = torch.zeros_like(shift_sm)
    for i in range(NS):
        if i == PILOT_0B:
            continue
        agc = ops_t.imageagc(image_masked[i], np_pad=30)
        refl_all[i] = shift_sm[i] * agc

    # ---------- Step 6: v_smooth for wavepath ----------
    v_s = ops_t.vel_smooth(v_t, nzs=11, nxs=21, niter=3)

    # ---------- Step 7: wavepath gradients (batched) ----------
    non_pilot = [i for i in range(NS) if i != PILOT_0B]
    gk1_tot = torch.zeros((NZ, NX), dtype=torch.float32, device=device)
    t1 = time.time()
    B = 8
    for b0 in range(0, len(non_pilot), B):
        sub = non_pilot[b0:b0 + B]
        sub_idx = torch.tensor(sub, device=device)
        res = pw1_wavepath_batch(
            v=v_s,
            refl=refl_all[sub_idx],
            dx=par.dx, dt=par.dt, nt=par.nt, npml=par.npml, src=src,
            p_ray=p_ray[sub_idx], xref_1=xref[sub_idx], zp1_phys=zp1[sub_idx],
            xg=xg[sub_idx], zg=zg[sub_idx], seis=seis_all[sub_idx],
            xs_taper_grid=40, dt_record=6,
        )
        # wavepath already normalises per-PW by max|img| → just weight by num[i]
        for k, i in enumerate(sub):
            gk1_tot = gk1_tot + res.img[k] * num[i]
    torch.cuda.synchronize()
    t_wp = time.time() - t1
    print(f"[T5/e2e] Wavepath 40 PW: {t_wp:.2f} s")

    # ---------- Step 8: dk1 = gk1 → top mute → gauss smooth ----------
    dk1 = gk1_tot.clone()
    dk1[:20, :] = 0.0
    dk1 = ops_t.gaussian_smooth_2d(2.0, 20.0, dk1)
    dk1_np = dk1.detach().to("cpu").numpy()

    # oracle
    grad_ref = read2d_np(rerun_dir / "gradient_1.bin", NZ, NX).astype(np.float32)
    p = pearson(grad_ref, dk1_np); r = rms_rel(grad_ref, dk1_np)
    print(f"[T5/e2e] gradient_1.bin: pearson={p:.6f}  rms_rel={r:.3e}  "
          f"ref_max={np.abs(grad_ref).max():.3e}  torch_max={np.abs(dk1_np).max():.3e}")
    print(f"[T5/e2e] Total FD time this run: {t_rtm + t_wp:.2f} s")
    assert p > 0.9999
    assert r < 0.02

    plot_2d_compare(grad_ref, dk1_np,
                    ["Fortran gradient_1.bin", "torch end-to-end", "diff"],
                    viz_dir / "t5_gradient_1.png",
                    clip_percentile=99.5)
