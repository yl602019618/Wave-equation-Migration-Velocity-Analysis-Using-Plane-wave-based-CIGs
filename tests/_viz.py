"""Minimal plotting helpers for parity visualisations (PNG only, no show)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_np(x):
    if hasattr(x, "detach"):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


def plot_2d_compare(a_np, b_np, titles, out_path: Path,
                    cmap="seismic", clip_percentile=99.0, figsize=(15, 4)):
    """Side-by-side 2-D field comparison: `a` vs `b` vs `(b - a)`.

    Each panel uses a symmetric colour scale derived from ``a_np`` (the first
    argument), which is normally the reference field.
    """
    a = _to_np(a_np).astype(np.float32)
    b = _to_np(b_np).astype(np.float32)
    diff = b - a
    vm = float(np.percentile(np.abs(a), clip_percentile) + 1e-30)
    vmd = float(np.percentile(np.abs(diff), clip_percentile) + 1e-30)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(a, aspect="auto", cmap=cmap, vmin=-vm, vmax=vm)
    axes[1].imshow(b, aspect="auto", cmap=cmap, vmin=-vm, vmax=vm)
    axes[2].imshow(diff, aspect="auto", cmap=cmap, vmin=-vmd, vmax=vmd)
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)
    fig.suptitle(f"max|a|={np.abs(a).max():.3e}  max|b|={np.abs(b).max():.3e}  "
                 f"rms|b-a|={np.sqrt((diff**2).mean()):.3e}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_traces(traces: dict, out_path: Path, figsize=(10, 5)):
    """Plot N 1-D curves labelled by dict keys."""
    fig, ax = plt.subplots(figsize=figsize)
    for lbl, y in traces.items():
        ax.plot(_to_np(y), label=lbl, linewidth=0.9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_2d_jet_compare(a_np, b_np, titles, out_path: Path, figsize=(15, 4)):
    """Compare fields with an absolute colour scale (e.g. velocity models).

    Uses a shared vmin/vmax from the first argument.
    """
    a = _to_np(a_np).astype(np.float32)
    b = _to_np(b_np).astype(np.float32)
    diff = b - a
    vmin, vmax = float(a.min()), float(a.max())
    vmd = float(np.percentile(np.abs(diff), 99.5) + 1e-30)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    im0 = axes[0].imshow(a, aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    im1 = axes[1].imshow(b, aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    im2 = axes[2].imshow(diff, aspect="auto", cmap="seismic", vmin=-vmd, vmax=vmd)
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.02)
    fig.colorbar(im1, ax=axes[1], fraction=0.04, pad=0.02)
    fig.colorbar(im2, ax=axes[2], fraction=0.04, pad=0.02)
    fig.suptitle(f"rms|b-a|={np.sqrt((diff**2).mean()):.3e}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def pearson(a_np, b_np) -> float:
    a = _to_np(a_np).ravel().astype(np.float64)
    b = _to_np(b_np).ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    na = np.sqrt((a * a).sum()); nb = np.sqrt((b * b).sum())
    if na == 0 or nb == 0:
        return 0.0
    return float((a * b).sum() / (na * nb))


def rms_rel(a_np, b_np) -> float:
    a = _to_np(a_np).astype(np.float64)
    b = _to_np(b_np).astype(np.float64)
    denom = np.sqrt((a * a).mean()) + 1e-30
    return float(np.sqrt(((b - a) ** 2).mean()) / denom)
