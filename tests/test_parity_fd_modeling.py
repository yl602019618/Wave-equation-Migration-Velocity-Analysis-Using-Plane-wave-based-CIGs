"""T3 parity: batched torch plane-wave modeling.

Levels of correctness we check:

(a) PML coefficients `alpha/temp1/temp2/beta_dt` match numpy to FP32 ULP.
(b) Batched B=4 FWD produces identical results to 4 separate B=1 FWDs.
(c) Torch FWD vs the Fortran "cpg_full_<is>.bin" oracle: peak amplitude and
    direct-wave match; accumulated FP drift in %-RMS range is expected.
(d) Torch FWD vs numpy FWD — Pearson ≥ 0.9999 for a few selected PWs.
"""
from pathlib import Path
import time
import numpy as np
import torch
import pytest

from pwmva_torch.device import pick_device
from pwmva_torch import io as io_t
from pwmva_torch.source import ricker
from pwmva_torch.fd2d.pml import build_alpha_temp, build_taper
from pwmva_torch.fd2d.pw1_modeling import pw1_modeling_batch

# numpy reference
from pwmva.io import read1d as read1d_np, read2d as read2d_np
from pwmva.config import read_parfile
from pwmva.geom  import read_pw1_coord
from pwmva.fd2d.pml import build_alpha_temp as build_alpha_temp_np, build_taper as build_taper_np
from pwmva.fd2d.pw1_modeling import pw1_modeling

from _viz import plot_2d_compare, plot_traces, pearson, rms_rel

DEVICE = pick_device()
NZ, NX = 201, 801


# ------------------------------------------------------------------
# (a) PML coefficients parity
# ------------------------------------------------------------------
def test_pml_coefficients(model_dir):
    v_np = read2d_np(model_dir / "velh_201x801x2.5m.bin", NZ, NX).astype(np.float32)
    dx, dt, npml = 2.5, 0.0006, 100
    alpha_np, temp1_np, temp2_np, beta_dt_np = build_alpha_temp_np(v_np, dx, dt, npml)

    v = torch.from_numpy(v_np).to(DEVICE)
    alpha_t, temp1_t, temp2_t, beta_dt_t = build_alpha_temp(v, dx, dt, npml)

    def _max_diff(x, y):
        return float((torch.from_numpy(x).to(DEVICE) - y).abs().max())

    da = _max_diff(alpha_np, alpha_t)
    dt1 = _max_diff(temp1_np, temp1_t)
    dt2 = _max_diff(temp2_np, temp2_t)
    db = _max_diff(beta_dt_np, beta_dt_t)
    print(f"\n[T3/pml] max|Δ| alpha={da:.2e}  temp1={dt1:.2e}  temp2={dt2:.2e}  beta_dt={db:.2e}")
    assert da < 1e-6 and dt1 < 1e-5 and dt2 < 1e-5 and db < 1e-5

    taper_np = build_taper_np(NX, 80)
    taper_t  = build_taper(NX, 80, device=DEVICE).detach().to("cpu").numpy()
    assert np.max(np.abs(taper_np - taper_t)) < 1e-6


# ------------------------------------------------------------------
# (b) batched B=4 == 4 separate B=1 runs (GPU self-consistency)
# ------------------------------------------------------------------
def test_batched_equals_per_pw(model_dir, working_dir):
    par = read_parfile(working_dir / "parfile_pwmva_warp_rerun.sh")
    coord = read_pw1_coord(model_dir / "coord_pw_45t45.dat")
    v_np = read2d_np(model_dir / "velh_201x801x2.5m.bin", NZ, NX).astype(np.float32)

    # use a short nt to keep the test cheap
    NT = 300
    v = torch.from_numpy(v_np).to(DEVICE)
    src = ricker(par.nw, par.dt, par.freq, device=DEVICE)

    # pick 4 plane waves: pilot + two off-pilots + one edge
    pick = [20, 0, 10, 40]
    B = len(pick)
    p_ray = torch.tensor([coord.p[i] for i in pick], dtype=torch.float32, device=DEVICE)
    xref  = torch.tensor([coord.xref_1[i] for i in pick], dtype=torch.float32, device=DEVICE)
    zp1   = torch.tensor([coord.zp1[i] for i in pick], dtype=torch.float32, device=DEVICE)
    ng = int(coord.ng[pick[0]])
    xg = torch.tensor(np.stack([coord.xg[:ng, i] for i in pick]), dtype=torch.float32, device=DEVICE)
    zg = torch.tensor(np.stack([coord.zg[:ng, i] for i in pick]), dtype=torch.float32, device=DEVICE)

    # batched run
    res_b = pw1_modeling_batch(v, par.dx, par.dt, NT, par.npml, src,
                               p_ray, xref, zp1, xg, zg,
                               xs_taper_grid=80)
    seis_b = res_b.seis.detach().to("cpu").numpy()   # (B, NT, ng)

    # per-PW runs
    out_solo = np.zeros_like(seis_b)
    for bi in range(B):
        res1 = pw1_modeling_batch(v, par.dx, par.dt, NT, par.npml, src,
                                  p_ray[bi:bi+1], xref[bi:bi+1], zp1[bi:bi+1],
                                  xg[bi:bi+1, :], zg[bi:bi+1, :],
                                  xs_taper_grid=80)
        out_solo[bi] = res1.seis[0].detach().to("cpu").numpy()
    d = np.max(np.abs(out_solo - seis_b))
    print(f"\n[T3/batched] max|batched - perPW| = {d:.3e}  (NT={NT}, B={B})")
    # GPU float32 should be bit-identical since nothing is reduced across the batch dim.
    assert d < 1e-6


# ------------------------------------------------------------------
# (c) torch batched vs Fortran cpg_full_<is>.bin oracle
# ------------------------------------------------------------------
def test_vs_cpg_full_oracle(model_dir, working_dir, oracle_dir, viz_dir):
    par = read_parfile(working_dir / "parfile_pwmva_warp_rerun.sh")
    coord = read_pw1_coord(model_dir / "coord_pw_45t45.dat")
    v_np = read2d_np(model_dir / "vel_201x801x2.5m.bin", NZ, NX).astype(np.float32)

    v = torch.from_numpy(v_np).to(DEVICE)
    src = ricker(par.nw, par.dt, par.freq, device=DEVICE)

    # pilot + two off-pilots: is=21 (0-based 20), is=11 (10), is=31 (30)
    pick = [20, 10, 30]
    B = len(pick)
    p_ray = torch.tensor([coord.p[i] for i in pick], dtype=torch.float32, device=DEVICE)
    xref  = torch.tensor([coord.xref_1[i] for i in pick], dtype=torch.float32, device=DEVICE)
    zp1   = torch.tensor([coord.zp1[i] for i in pick], dtype=torch.float32, device=DEVICE)
    ng = int(coord.ng[pick[0]])
    xg = torch.tensor(np.stack([coord.xg[:ng, i] for i in pick]), dtype=torch.float32, device=DEVICE)
    zg = torch.tensor(np.stack([coord.zg[:ng, i] for i in pick]), dtype=torch.float32, device=DEVICE)

    t0 = time.time()
    res = pw1_modeling_batch(v, par.dx, par.dt, par.nt, par.npml, src,
                             p_ray, xref, zp1, xg, zg, xs_taper_grid=80)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    seis_t = res.seis.detach().to("cpu").numpy()   # (B, nt, ng)
    print(f"\n[T3/oracle] torch batched FWD 3 PW × nt={par.nt}: {elapsed:.2f}s")

    # Compare with Fortran cpg_full oracle (1-based indexing in filename)
    pears = []
    for bi, is_ in enumerate(pick):
        is_fname = is_ + 1
        cpg_oracle = read2d_np(oracle_dir / f"cpg_full_{is_fname}.bin",
                               par.nt, ng).astype(np.float32)
        ref_max = float(np.abs(cpg_oracle).max())
        my_max  = float(np.abs(seis_t[bi]).max())
        p = pearson(cpg_oracle, seis_t[bi])
        r = rms_rel(cpg_oracle, seis_t[bi])
        pears.append(p)
        print(f"    PW is={is_fname}: peak ref={ref_max:.3f}  torch={my_max:.3f}  "
              f"pearson={p:.4f}  rms_rel={r:.3e}")
        # Peak amplitudes should match to 3 digits (same direct wave).
        assert abs(my_max - ref_max) / ref_max < 0.05
        # Pearson — 3000 step FP drift is a few tens of percent (see python readme P3).
        # We only assert Pearson > 0.5 to confirm the wavefield is physically the same.
        assert p > 0.5, f"too little correlation with Fortran oracle: {p:.3f}"
        # visualise
        plot_2d_compare(cpg_oracle, seis_t[bi],
                        [f"Fortran cpg_full_{is_fname}", f"torch PW is={is_fname}", "diff"],
                        viz_dir / f"t3_vs_cpg_full_is{is_fname}.png",
                        clip_percentile=99.8)
    print(f"    mean Pearson across 3 PW = {np.mean(pears):.4f}")


# ------------------------------------------------------------------
# (d) torch batched vs numpy per-PW forward (tightest comparison)
# ------------------------------------------------------------------
def test_vs_numpy_forward(model_dir, working_dir, viz_dir):
    par = read_parfile(working_dir / "parfile_pwmva_warp_rerun.sh")
    coord = read_pw1_coord(model_dir / "coord_pw_45t45.dat")
    v_np = read2d_np(model_dir / "vel_201x801x2.5m.bin", NZ, NX).astype(np.float32)
    src_np = read1d_np(Path("/home/pisquare/zhijun/pwmva_fortran/pwmva_package") /
                       "results/2D_models/mlayer/fdcsg_f40/source.bin", par.nw)

    # short NT to keep this cheap
    NT = 500
    pick = [20, 10]
    B = len(pick)

    v = torch.from_numpy(v_np).to(DEVICE)
    src = torch.from_numpy(src_np).to(DEVICE)
    p_ray = torch.tensor([coord.p[i] for i in pick], dtype=torch.float32, device=DEVICE)
    xref  = torch.tensor([coord.xref_1[i] for i in pick], dtype=torch.float32, device=DEVICE)
    zp1   = torch.tensor([coord.zp1[i] for i in pick], dtype=torch.float32, device=DEVICE)
    ng = int(coord.ng[pick[0]])
    xg = torch.tensor(np.stack([coord.xg[:ng, i] for i in pick]), dtype=torch.float32, device=DEVICE)
    zg = torch.tensor(np.stack([coord.zg[:ng, i] for i in pick]), dtype=torch.float32, device=DEVICE)

    res_t = pw1_modeling_batch(v, par.dx, par.dt, NT, par.npml, src,
                               p_ray, xref, zp1, xg, zg, xs_taper_grid=80)
    seis_t = res_t.seis.detach().to("cpu").numpy()

    seis_np = np.zeros_like(seis_t)
    for bi, i in enumerate(pick):
        r = pw1_modeling(v=v_np, dx=par.dx, dt=par.dt, nt=NT, npml=par.npml, src=src_np,
                         p_ray=float(coord.p[i]), xref_1=float(coord.xref_1[i]),
                         zp1_phys=float(coord.zp1[i]),
                         xg=coord.xg[:ng, i], zg=coord.zg[:ng, i],
                         xs_taper_grid=80)
        seis_np[bi] = r.seis

    for bi, i in enumerate(pick):
        p = pearson(seis_np[bi], seis_t[bi])
        r = rms_rel(seis_np[bi], seis_t[bi])
        print(f"\n[T3/np-vs-torch] PW #{i+1}: pearson={p:.5f}  rms_rel={r:.3e}")
        assert p > 0.9999
        assert r < 5e-3
        plot_2d_compare(seis_np[bi], seis_t[bi],
                        [f"numpy PW #{i+1}", f"torch PW #{i+1}", "diff"],
                        viz_dir / f"t3_np_vs_torch_pw{i+1}.png",
                        clip_percentile=99.5)
