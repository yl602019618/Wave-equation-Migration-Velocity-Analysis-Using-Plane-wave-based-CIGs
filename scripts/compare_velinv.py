"""Compare torch inversion output (velinv_<N>.bin, gradient_<N>.bin) against
both the Fortran rerun and numpy parity run.

Usage:
  python scripts/compare_velinv.py --iter 1 --out /tmp/torch_iter1
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from _viz import plot_2d_compare, plot_2d_jet_compare, pearson, rms_rel

from pwmva.io import read2d as read2d_np

PKG = Path("/home/pisquare/zhijun/pwmva_fortran/pwmva_package")
RERUN = PKG / "results/2D_models/mlayer/pwmva_warp_rerun"
NZ, NX = 201, 801


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--viz-out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "tests" / "_viz_out")
    args = ap.parse_args()
    args.viz_out.mkdir(exist_ok=True)

    v_torch = read2d_np(args.out / f"velinv_{args.iter}.bin", NZ, NX)
    v_ref   = read2d_np(RERUN / f"velinv_{args.iter}.bin",  NZ, NX)
    v_true  = read2d_np(PKG / "model/2D_models/mlayer/vel_201x801x2.5m.bin", NZ, NX)
    v_init  = read2d_np(PKG / "model/2D_models/mlayer/velh_201x801x2.5m.bin", NZ, NX)

    print(f"\n== velinv_{args.iter}.bin ==")
    rms = np.sqrt(((v_torch - v_ref) ** 2).mean())
    mxd = np.abs(v_torch - v_ref).max()
    p = pearson(v_ref, v_torch)
    print(f"  torch vs Fortran rerun: RMS = {rms:.3f} m/s   max|Δ| = {mxd:.3f}   Pearson = {p:.6f}")
    rms_true = np.sqrt(((v_torch - v_true) ** 2).mean())
    rms_true_ref = np.sqrt(((v_ref - v_true) ** 2).mean())
    print(f"  torch vs TRUE         : RMS = {rms_true:.3f} m/s")
    print(f"  Fortran vs TRUE       : RMS = {rms_true_ref:.3f} m/s")
    print(f"  velh  vs TRUE         : RMS = {np.sqrt(((v_init - v_true)**2).mean()):.3f} m/s")

    # gradient
    try:
        g_torch = read2d_np(args.out / f"gradient_{args.iter}.bin", NZ, NX)
        g_ref   = read2d_np(RERUN / f"gradient_{args.iter}.bin",  NZ, NX)
        p_g = pearson(g_ref, g_torch); r_g = rms_rel(g_ref, g_torch)
        print(f"\n== gradient_{args.iter}.bin ==")
        print(f"  torch vs Fortran: Pearson = {p_g:.6f}   rms_rel = {r_g:.3e}")
    except FileNotFoundError:
        g_torch = g_ref = None

    # --- 6-panel figure: true, Fortran rerun, torch, diff, gradient torch vs ref
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    vmin, vmax = float(v_true.min()), float(v_true.max())
    axes[0, 0].imshow(v_init, aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"velh (init)   RMS vs true = "
                         f"{np.sqrt(((v_init - v_true)**2).mean()):.1f}")
    axes[0, 1].imshow(v_ref, aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"Fortran rerun velinv_{args.iter}   "
                         f"RMS vs true = {rms_true_ref:.1f}")
    axes[0, 2].imshow(v_torch, aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f"torch velinv_{args.iter}   RMS vs Fortran = {rms:.2f}")

    diff_vt = v_torch - v_true
    diff_vf = v_torch - v_ref
    vmd1 = float(np.abs(diff_vt).max()) + 1e-30
    vmd2 = float(np.abs(diff_vf).max()) + 1e-30
    axes[1, 0].imshow(v_true, aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("TRUE velocity")
    axes[1, 1].imshow(diff_vt, aspect="auto", cmap="seismic", vmin=-vmd1, vmax=vmd1)
    axes[1, 1].set_title(f"torch - TRUE  (max|Δ|={vmd1:.1f})")
    axes[1, 2].imshow(diff_vf, aspect="auto", cmap="seismic", vmin=-vmd2, vmax=vmd2)
    axes[1, 2].set_title(f"torch - Fortran rerun  (max|Δ|={vmd2:.3f})")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"iter {args.iter}  —  torch vs Fortran rerun vs TRUE", fontsize=12)
    fig.tight_layout()
    out = args.viz_out / f"t6_iter{args.iter}_six_panel.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {out}")

    if g_torch is not None:
        plot_2d_compare(g_ref, g_torch,
                        [f"Fortran gradient_{args.iter}", "torch", "diff"],
                        args.viz_out / f"t6_gradient_iter{args.iter}.png",
                        clip_percentile=99.5)
        print(f"Figure: {args.viz_out / f't6_gradient_iter{args.iter}.png'}")


if __name__ == "__main__":
    main()
