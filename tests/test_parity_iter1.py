"""T6 parity: full iter-1 inversion torch vs Fortran rerun.

This test reads a pre-run ``velinv_1.bin`` from the directory given by
``PWMVA_TORCH_ITER1_OUT`` (default /tmp/torch_iter1) and compares it to the
Fortran rerun oracle. We do NOT run the full iteration inside pytest (too slow
for CI) — run ``scripts/run_iter.py --niter 1 --out /tmp/torch_iter1`` first.
"""
import os
from pathlib import Path
import numpy as np
import pytest

from pwmva.io import read2d as read2d_np

from _viz import plot_2d_compare, plot_2d_jet_compare, pearson, rms_rel

NZ, NX = 201, 801
OUT = Path(os.environ.get("PWMVA_TORCH_ITER1_OUT", "/tmp/torch_iter1"))


@pytest.mark.skipif(not (OUT / "velinv_1.bin").exists(),
                    reason="run scripts/run_iter.py --niter 1 --out {OUT} first")
def test_velinv_1_vs_fortran(rerun_dir, viz_dir):
    v_t   = read2d_np(OUT / "velinv_1.bin", NZ, NX)
    v_ref = read2d_np(rerun_dir / "velinv_1.bin", NZ, NX)

    rms = np.sqrt(((v_t - v_ref) ** 2).mean())
    mxd = np.abs(v_t - v_ref).max()
    p   = pearson(v_ref, v_t)
    print(f"\n[T6] velinv_1.bin: RMS = {rms:.3f} m/s  max|Δ| = {mxd:.3f}  Pearson = {p:.6f}")
    # plan target: < 1 m/s
    assert rms < 1.0
    assert p > 0.999999

    plot_2d_jet_compare(v_ref, v_t, ["Fortran rerun velinv_1", "torch velinv_1", "diff"],
                        viz_dir / "t6_velinv_1_jet.png")


@pytest.mark.skipif(not (OUT / "gradient_1.bin").exists(),
                    reason="run scripts/run_iter.py first")
def test_gradient_1_vs_fortran(rerun_dir, viz_dir):
    g_t   = read2d_np(OUT / "gradient_1.bin", NZ, NX)
    g_ref = read2d_np(rerun_dir / "gradient_1.bin", NZ, NX)
    p = pearson(g_ref, g_t); r = rms_rel(g_ref, g_t)
    print(f"\n[T6] gradient_1.bin: Pearson = {p:.6f}  rms_rel = {r:.3e}")
    assert p > 0.9999
    assert r < 1e-2
    plot_2d_compare(g_ref, g_t,
                    ["Fortran gradient_1", "torch gradient_1", "diff"],
                    viz_dir / "t6_gradient_1.png",
                    clip_percentile=99.5)
