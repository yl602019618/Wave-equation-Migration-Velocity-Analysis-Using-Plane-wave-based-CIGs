"""T1 parity: Ricker wavelet (torch) vs Fortran ``source.bin`` and numpy port."""
from pathlib import Path
import numpy as np
import torch
import pytest

from pwmva_torch.source import ricker as ricker_t
from pwmva_torch.device import pick_device

# numpy reference (in pwmva_python on sys.path via conftest)
from pwmva.source import ricker as ricker_np
from pwmva.io import read1d as read1d_np

from _viz import plot_traces


def test_ricker_vs_source_bin(csg_dir, viz_dir):
    nw, dt, freq = 3000, 0.0006, 40.0
    device = pick_device()

    src_bin = read1d_np(csg_dir / "source.bin", nw)         # numpy (3000,)
    s_np    = ricker_np(nw, dt, freq)
    s_cpu   = ricker_t(nw, dt, freq, device="cpu").numpy()
    s_gpu   = ricker_t(nw, dt, freq, device=device).detach().to("cpu").numpy()

    d_vs_bin = np.max(np.abs(s_gpu - src_bin))
    d_vs_np  = np.max(np.abs(s_gpu - s_np))
    d_cpu_gpu = np.max(np.abs(s_gpu - s_cpu))
    print(f"\n[T1] max|torch_gpu - source.bin| = {d_vs_bin:.2e}")
    print(f"[T1] max|torch_gpu - numpy|       = {d_vs_np:.2e}")
    print(f"[T1] max|torch_gpu - torch_cpu|   = {d_cpu_gpu:.2e}")

    # Fortran bin: 1e-6 tolerance (float32 ULP)
    assert d_vs_bin < 1e-6
    # vs numpy port: should match to float32 precision of per-op reordering
    assert d_vs_np < 1e-6
    assert d_cpu_gpu < 1e-6

    plot_traces({"source.bin": src_bin, "torch (gpu)": s_gpu,
                 "diff × 1e6": (s_gpu - src_bin) * 1e6},
                viz_dir / "t1_ricker_vs_source_bin.png")
