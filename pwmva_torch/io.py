"""Binary I/O matching Fortran ``write_binfile`` (raw little-endian float32).

Fortran writes ``a(nz, nx)`` column-major → on disk layout matches NumPy
``(nx, nz)`` C-order → we read + transpose to ``(nz, nx)`` and return a torch
tensor. Mirrors ``pwmva_python.io`` but returns ``torch.Tensor`` on the
requested device.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from .device import DTYPE

NP_DTYPE = np.dtype("<f4")


def read2d(path: str | Path, nz: int, nx: int,
           device: torch.device | str | None = None) -> torch.Tensor:
    a = np.fromfile(path, dtype=NP_DTYPE)
    if a.size != nz * nx:
        raise ValueError(f"{path}: expected {nz*nx} f32, got {a.size}")
    arr = a.reshape(nx, nz).T.copy()
    return torch.from_numpy(arr).to(dtype=DTYPE, device=device) if device is not None \
        else torch.from_numpy(arr).to(dtype=DTYPE)


def read3d(path: str | Path, nz: int, nx: int, ns: int,
           device: torch.device | str | None = None) -> torch.Tensor:
    a = np.fromfile(path, dtype=NP_DTYPE)
    if a.size != nz * nx * ns:
        raise ValueError(f"{path}: expected {nz*nx*ns} f32, got {a.size}")
    arr = a.reshape(ns, nx, nz).transpose(2, 1, 0).copy()
    return torch.from_numpy(arr).to(dtype=DTYPE, device=device) if device is not None \
        else torch.from_numpy(arr).to(dtype=DTYPE)


def read1d(path: str | Path, n: int,
           device: torch.device | str | None = None) -> torch.Tensor:
    a = np.fromfile(path, dtype=NP_DTYPE)
    if a.size != n:
        raise ValueError(f"{path}: expected {n} f32, got {a.size}")
    return torch.from_numpy(a.copy()).to(dtype=DTYPE, device=device) if device is not None \
        else torch.from_numpy(a.copy()).to(dtype=DTYPE)


def write2d(path: str | Path, a: torch.Tensor) -> None:
    if a.ndim != 2:
        raise ValueError("write2d expects (nz, nx)")
    arr = a.detach().to("cpu", DTYPE).numpy()
    arr.T.astype(NP_DTYPE, copy=False).tofile(path)


def write3d(path: str | Path, a: torch.Tensor) -> None:
    if a.ndim != 3:
        raise ValueError("write3d expects (nz, nx, ns)")
    arr = a.detach().to("cpu", DTYPE).numpy()
    arr.transpose(2, 1, 0).astype(NP_DTYPE, copy=False).tofile(path)


def write1d(path: str | Path, a: torch.Tensor) -> None:
    arr = a.detach().to("cpu", DTYPE).numpy()
    arr.astype(NP_DTYPE, copy=False).tofile(path)
