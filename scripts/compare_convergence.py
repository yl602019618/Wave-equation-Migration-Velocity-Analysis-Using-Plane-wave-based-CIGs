"""Compare torch multi-iteration convergence against Fortran + (optional) Python.

Produces:
  - Table of per-iter (misfit, fff, RMS vs Fortran, RMS vs TRUE)
  - ``t7_convergence.png`` : RMS vs TRUE / RMS vs Fortran across iterations
  - ``t7_velinv_evolution.png`` : side-by-side velinv for iters [1, N/2, N]
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
from _viz import pearson, rms_rel

from pwmva.io import read2d as read2d_np

PKG = Path("/home/pisquare/zhijun/pwmva_fortran/pwmva_package")
RERUN = PKG / "results/2D_models/mlayer/pwmva_warp_rerun"
NZ, NX = 201, 801


def _safe_read(path: Path):
    if not path.exists():
        return None
    return read2d_np(path, NZ, NX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",    type=Path, required=True, help="torch output dir")
    ap.add_argument("--py-out", type=Path, default=None,  help="python output dir (optional)")
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--viz-out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "tests" / "_viz_out")
    args = ap.parse_args()
    args.viz_out.mkdir(exist_ok=True)

    v_true = read2d_np(PKG / "model/2D_models/mlayer/vel_201x801x2.5m.bin", NZ, NX)
    v_init = read2d_np(PKG / "model/2D_models/mlayer/velh_201x801x2.5m.bin", NZ, NX)

    rows = []
    it_list = []
    for k in range(1, args.max_iter + 1):
        v_t = _safe_read(args.out / f"velinv_{k}.bin")
        if v_t is None:
            break
        v_f = _safe_read(RERUN / f"velinv_{k}.bin")
        v_p = _safe_read(args.py_out / f"velinv_{k}.bin") if args.py_out else None

        rms_vs_f = np.sqrt(((v_t - v_f) ** 2).mean()) if v_f is not None else float("nan")
        rms_vs_t = np.sqrt(((v_t - v_true) ** 2).mean())
        f_vs_t = np.sqrt(((v_f - v_true) ** 2).mean()) if v_f is not None else float("nan")
        p_vs_f = pearson(v_f, v_t) if v_f is not None else float("nan")
        rms_vs_py = np.sqrt(((v_t - v_p) ** 2).mean()) if v_p is not None else float("nan")
        rows.append((k, rms_vs_f, rms_vs_t, f_vs_t, p_vs_f, rms_vs_py))
        it_list.append(k)

    print(f"{'iter':>4}  {'RMS vs Fortran':>14}  {'RMS vs TRUE':>11}  "
          f"{'Fortran vs TRUE':>15}  {'Pearson':>9}"
          + ("  RMS vs Python" if args.py_out else ""))
    for k, rf, rt, ft, pf, rp in rows:
        extra = f"  {rp:>13.3f}" if args.py_out else ""
        print(f"{k:>4}  {rf:>14.4f}  {rt:>11.3f}  {ft:>15.3f}  {pf:>9.6f}" + extra)

    # --- convergence plot
    iters = np.array(it_list)
    rms_torch_true = np.array([r[2] for r in rows])
    rms_fort_true  = np.array([r[3] for r in rows])
    rms_torch_fort = np.array([r[1] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(iters, rms_torch_true, "o-", label="torch vs TRUE", color="C0")
    axes[0].plot(iters, rms_fort_true,  "s-", label="Fortran vs TRUE", color="C1", alpha=0.8)
    axes[0].axhline(np.sqrt(((v_init - v_true) ** 2).mean()),
                    color="gray", linestyle="--", label="velh init vs TRUE")
    axes[0].set_xlabel("iter"); axes[0].set_ylabel("RMS velocity (m/s)")
    axes[0].set_title("Convergence vs true model")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(iters, rms_torch_fort, "o-", color="C2")
    axes[1].set_xlabel("iter"); axes[1].set_ylabel("RMS torch-Fortran (m/s)")
    axes[1].set_title("Inter-implementation FP drift")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = args.viz_out / "t7_convergence.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\nSaved {out}")

    # --- evolution 3-panel (iters 1, mid, last)
    N = len(rows)
    if N >= 1:
        picks = sorted(set([1, max(1, N // 2), N]))
        vs = [(_safe_read(args.out / f"velinv_{k}.bin"), k) for k in picks]
        fig, axes = plt.subplots(1, len(vs) + 1, figsize=(4.5 * (len(vs) + 1), 4))
        vmin, vmax = float(v_true.min()), float(v_true.max())
        axes[0].imshow(v_true, aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
        axes[0].set_title("TRUE")
        for ax, (v, k) in zip(axes[1:], vs):
            rms = np.sqrt(((v - v_true) ** 2).mean())
            ax.imshow(v, aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
            ax.set_title(f"torch iter {k}  RMS={rms:.1f}")
        for ax in axes: ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        out = args.viz_out / "t7_velinv_evolution.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
