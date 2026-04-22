"""Image-domain operators (torch versions).

Line-by-line ports of ``pwmva_python.image_ops``. Each op accepts either a 2-D
``(nz, nx)`` tensor or a 3-D ``(..., nz, nx)`` tensor where leading dims are
treated as "batch / plane-wave" axes and processed independently.

All arithmetic in float32 to stay comparable with the Fortran oracle.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from .device import DTYPE

# 4th-order central FD coefficients (module_global.f90)
CF41 = 2.0 / 3.0
CF42 = -1.0 / 12.0


# ----------------------------------------------------------------------
# z derivative
# ----------------------------------------------------------------------
def z_derivative_2d(f: torch.Tensor, fd_order: int = 4) -> torch.Tensor:
    """Central z-derivative along last-but-one axis. Supports batched input.

    Replicates Fortran edge behaviour: first/last ``fd_order//2`` rows repeat
    the nearest valid row value.
    """
    if fd_order not in (2, 4):
        raise NotImplementedError(f"fd_order={fd_order}")
    nz = f.shape[-2]
    out = torch.zeros_like(f)
    t = fd_order // 2
    if fd_order == 4:
        out[..., t:nz - t, :] = (
            CF41 * (f[..., t + 1:nz - t + 1, :] - f[..., t - 1:nz - t - 1, :])
            + CF42 * (f[..., t + 2:nz - t + 2, :] - f[..., t - 2:nz - t - 2, :])
        )
    else:  # fd_order == 2
        out[..., t:nz - t, :] = 0.5 * (f[..., t + 1:nz - t + 1, :]
                                       - f[..., t - 1:nz - t - 1, :])
    out[..., :t, :] = out[..., t:t + 1, :]
    out[..., nz - t:, :] = out[..., nz - t - 1:nz - t, :]
    return out


# ----------------------------------------------------------------------
# Gaussian smoothing (separable, un-normalised kernel — match Fortran)
# ----------------------------------------------------------------------
def _gaussian_kernel(n: int, sigma: float,
                     device=None, dtype=DTYPE) -> torch.Tensor:
    """Unnormalised Gaussian: g(i) = exp(-((i - ceil(n/2))/sigma)^2 / 2), i=1..n."""
    i = torch.arange(1, n + 1, dtype=dtype, device=device)
    c = math.ceil(n / 2.0)
    return torch.exp(-((i - c) / sigma) ** 2 / 2.0)


def _conv_gauss_1d_batch(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Linear conv of ``x`` (..., n) with ``g`` (n,) via FFT; output length n.

    Matches Fortran ``conv_gauss``: full linear conv length 2n-1, then crop
    ``[ceil(n/2)-1 : ceil(n/2)-1 + n]``.
    """
    n = x.shape[-1]
    L = 2 * n - 1
    # zero-pad both to L and convolve via rfft for speed
    X = torch.fft.rfft(x, n=L)
    G = torch.fft.rfft(g, n=L)
    y = torch.fft.irfft(X * G, n=L)
    start = int(math.ceil(n / 2.0)) - 1
    return y[..., start:start + n]


def gaussian_smooth_2d(sigmaz: float, sigmax: float,
                       h: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian smoothing (x then z). Supports ``(..., nz, nx)``.

    Kernels are un-normalised (Fortran convention) → result scales by ∏sum(g).
    """
    nz, nx = h.shape[-2], h.shape[-1]
    device, dtype = h.device, h.dtype
    gx = _gaussian_kernel(nx, sigmax, device=device, dtype=dtype)
    gz = _gaussian_kernel(nz, sigmaz, device=device, dtype=dtype)
    # conv along x axis (last)
    out = _conv_gauss_1d_batch(h, gx)
    # conv along z axis (second-to-last) — transpose, conv, transpose back
    out = out.transpose(-1, -2).contiguous()
    out = _conv_gauss_1d_batch(out, gz)
    return out.transpose(-1, -2).contiguous()


# ----------------------------------------------------------------------
# vel_smooth: boxcar on slowness via 2-D prefix sum (integral image)
# ----------------------------------------------------------------------
def _moving_avg_2d(d: torch.Tensor, n1p: int, n2p: int) -> torch.Tensor:
    """Replicate-padded (n1p × n2p) box average — odd sizes.

    Uses 2-D prefix sum so a ``B × nz × nx`` input is still O(B·nz·nx).
    """
    h1 = (n1p - 1) // 2
    h2 = (n2p - 1) // 2
    # F.pad replicate requires 3-D or 4-D input → ensure at least 4-D
    orig_shape = d.shape
    if d.ndim == 2:
        d4 = d.unsqueeze(0).unsqueeze(0)
    elif d.ndim == 3:
        d4 = d.unsqueeze(1)
    else:
        d4 = d
    pad = F.pad(d4, (h2, h2, h1, h1), mode="replicate").to(torch.float64)
    pad = pad.view(-1, pad.shape[-2], pad.shape[-1]) if d.ndim >= 3 else pad.squeeze(0).squeeze(0)
    # prefix sum with leading zero row/col
    if pad.ndim == 2:
        nz_pad, nx_pad = pad.shape
        P = torch.zeros(nz_pad + 1, nx_pad + 1, dtype=torch.float64, device=pad.device)
        P[1:, 1:] = pad.cumsum(0).cumsum(1)
        nz = orig_shape[-2]; nx = orig_shape[-1]
        s = (P[n1p:n1p + nz, n2p:n2p + nx]
             - P[:nz,           n2p:n2p + nx]
             - P[n1p:n1p + nz, :nx]
             + P[:nz,           :nx])
    else:
        # batched case (B, nz_pad, nx_pad)
        B, nz_pad, nx_pad = pad.shape
        P = torch.zeros(B, nz_pad + 1, nx_pad + 1,
                        dtype=torch.float64, device=pad.device)
        P[:, 1:, 1:] = pad.cumsum(-2).cumsum(-1)
        nz = orig_shape[-2]; nx = orig_shape[-1]
        s = (P[:, n1p:n1p + nz, n2p:n2p + nx]
             - P[:, :nz,           n2p:n2p + nx]
             - P[:, n1p:n1p + nz, :nx]
             + P[:, :nz,           :nx])
        s = s.view(*orig_shape[:-2], nz, nx)
    return (s / float(n1p * n2p)).to(d.dtype)


def vel_smooth(vin: torch.Tensor, nzs: int, nxs: int, niter: int) -> torch.Tensor:
    """Smooth slowness 1/v `niter` times with (nzs × nxs) box, then invert."""
    s = 1.0 / vin.to(DTYPE)
    for _ in range(niter):
        s = _moving_avg_2d(s, nzs, nxs)
    return (1.0 / s).to(DTYPE)


# ----------------------------------------------------------------------
# mask routines
# ----------------------------------------------------------------------
def maskimage(img: torch.Tensor, theta: float, eps: int) -> torch.Tensor:
    """Zero a left/right wedge at every depth: bb(iz) = ceil(iz·|tan θ|) + eps."""
    out = img.clone()
    nz, nx = out.shape[-2], out.shape[-1]
    iz_1b = torch.arange(1, nz + 1, dtype=DTYPE, device=out.device)
    bb = torch.ceil(iz_1b * abs(math.tan(theta))).to(torch.int32) + int(eps)
    bb = torch.clamp(bb, max=nx)
    # build a boolean mask (nz, nx): True where col is inside the wedge
    cols = torch.arange(nx, device=out.device).unsqueeze(0)          # (1, nx)
    left = cols < bb.unsqueeze(1)                                    # (nz, nx)
    right = cols >= (nx - bb).unsqueeze(1)                           # (nz, nx)
    mask = left | right
    out = out.masked_fill(mask, 0.0)
    return out


def snell_mute(img: torch.Tensor, p: float, slowness: torch.Tensor,
               tol: float = 5e-5) -> torch.Tensor:
    """Per-column: find first depth where |p|+tol ≥ slowness[iz,ix], zero below."""
    out = img.clone()
    nz, nx = out.shape[-2], out.shape[-1]
    cond = (abs(p) + tol) >= slowness                                # (nz, nx)
    any_hit = cond.any(dim=0)                                        # (nx,)
    # argmax on bool along dim 0 gives first True (or 0 if none)
    first = cond.to(torch.int32).argmax(dim=0)                       # (nx,)
    first = torch.where(any_hit, first, torch.full_like(first, nz))
    row_idx = torch.arange(nz, device=out.device).unsqueeze(1)       # (nz, 1)
    mask = row_idx >= first.unsqueeze(0)                             # (nz, nx)
    return out.masked_fill(mask, 0.0)


def top_mute(img: torch.Tensor, top: int = 30) -> torch.Tensor:
    out = img.clone()
    out[..., :top, :] = 0.0
    return out


# ----------------------------------------------------------------------
# AGC
# ----------------------------------------------------------------------
def _normalize2d(d: torch.Tensor) -> torch.Tensor:
    m = d.abs().max()
    return d if float(m) == 0.0 else (d / m).to(DTYPE)


def imageagc(image: torch.Tensor, np_pad: int) -> torch.Tensor:
    """Port of ``imageagc(image, n1, n2, np)``.

    Top/bottom padded by ``np_pad`` rows mirrored across x; square + box-conv
    along z length ``np_pad``; crop; normalise; ``image/(sqrt(envelope)+1e-4)``;
    normalise again.
    """
    n1, n2 = image.shape
    device, dtype = image.device, image.dtype
    n11 = n1 + 2 * np_pad

    temp = torch.empty((n11, n2), dtype=dtype, device=device)
    # top: image[:np_pad, :] with x reversed
    temp[:np_pad, :] = image[:np_pad, :].flip(-1)
    temp[np_pad:np_pad + n1, :] = image
    temp[np_pad + n1:, :] = image[n1 - np_pad:n1, :].flip(-1)
    temp = temp ** 2

    # 1-D boxcar of length np_pad along z, computed as linear conv via FFT
    # (matches numpy scipy.signal.fftconvolve "full" mode ordering).
    kernel = torch.full((np_pad,), 1.0 / np_pad, dtype=dtype, device=device)
    L = n11 + np_pad - 1
    X = torch.fft.rfft(temp, n=L, dim=0)
    K = torch.fft.rfft(kernel, n=L)
    full = torch.fft.irfft(X * K.unsqueeze(-1), n=L, dim=0)   # (L, n2)
    start = np_pad + np_pad // 2
    out = full[start:start + n1, :]
    out = _normalize2d(out)
    out = image / (out.abs().sqrt() + 1e-4)
    out = _normalize2d(out)
    return out


# ----------------------------------------------------------------------
# scalar reductions
# ----------------------------------------------------------------------
def mean_abs(d: torch.Tensor) -> float:
    return float(d.abs().mean())


def nzmean2d(d: torch.Tensor) -> float:
    nz_count = int((d != 0.0).sum())
    if nz_count == 0:
        return 0.0
    return float(d.abs().sum()) / nz_count
