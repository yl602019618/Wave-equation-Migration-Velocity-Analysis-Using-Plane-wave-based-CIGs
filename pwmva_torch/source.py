"""Ricker source wavelet — port of ``module_source.f90::ricker``.

Identical math to ``pwmva_python.source.ricker`` but returns ``torch.Tensor``.
Amplitude normalised to ±0.5 (Fortran: ``s = -amp / (2 * max|amp|)``).
"""
from __future__ import annotations
import math
import torch

from .device import DTYPE


def ricker(nw: int, dt: float, freq: float,
           device: torch.device | str | None = None) -> torch.Tensor:
    tshift = math.sqrt(2.0) / freq
    pi2    = math.sqrt(math.pi) / 2.0
    b      = math.sqrt(6.0) / (math.pi * freq)
    const  = 2.0 * math.sqrt(6.0) / b

    t = torch.arange(nw, dtype=DTYPE, device=device) * dt
    u = const * (t - tshift)
    amp = ((u * u) / 4.0 - 0.5) * pi2 * torch.exp(-u * u / 4.0)
    s = -amp
    smax = s.abs().max() * 2.0
    return (s / smax).to(DTYPE)
