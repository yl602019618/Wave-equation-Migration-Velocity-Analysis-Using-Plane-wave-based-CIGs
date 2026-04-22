"""PML damping + edge taper — torch port of ``pwmva_python.fd2d.pml``.

Produces per-cell coefficients ``(alpha, temp1, temp2, beta_dt)`` used by the
batched stencil. All tensors live on the requested device in float32.

Defaults match the Fortran configuration for ``pwmva_warp_rerun``:
  FD_TYPE=2ND  FD_ORDER=28  BC_TYPE=1 (ABC)  IsFS=.false.
"""
from __future__ import annotations
import math
import numpy as np
import torch

from ..device import DTYPE

# 8th-order Laplacian stencil coefficients (module_global.f90)
C81 = -205.0 / 72.0
C82 =    8.0 /   5.0
C83 =   -1.0 /   5.0
C84 =    8.0 / 315.0
C85 =   -1.0 / 560.0


def abc_get_damp2d(nx: int, nz: int, npml: int, dx: float, cmin: float,
                   device: torch.device | str | None = None) -> torch.Tensor:
    """Quadratic PML damping, identical to Fortran ``abc_get_damp2d``."""
    nz_pml = nz + 2 * npml
    nx_pml = nx + 2 * npml
    a = (npml - 1) * dx
    kappa = 3.0 * cmin * math.log(1e7) / (2.0 * a)
    xa = np.arange(npml, dtype=np.float64) * dx / a
    damp1d = (kappa * xa * xa).astype(np.float32)

    damp = np.zeros((nz_pml, nx_pml), dtype=np.float32)
    # left/right walls (full height)
    for ix in range(npml):
        damp[:, npml - ix - 1]    = damp1d[ix]
        damp[:, nx + npml + ix]   = damp1d[ix]
    # top/bottom walls — Fortran restricts ix range as a triangle
    for iz in range(npml):
        x_lo = npml - iz
        x_hi = nx + npml + iz           # exclusive end
        damp[npml - iz - 1, x_lo - 1:x_hi] = damp1d[iz]
        damp[nz + npml + iz, x_lo - 1:x_hi] = damp1d[iz]
    return torch.from_numpy(damp).to(device=device, dtype=DTYPE)


def build_alpha_temp(v: torch.Tensor, dx: float, dt: float, npml: int
                     ) -> tuple[torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor]:
    """Pad velocity with replicate edge, build alpha, temp1, temp2, beta_dt.

    Parameters
    ----------
    v : torch.Tensor  (nz, nx) float32, m/s (on any device)

    Returns
    -------
    alpha, temp1, temp2, beta_dt : torch.Tensor  (nz_pml, nx_pml) float32, same device
    """
    v = v.to(DTYPE)
    device = v.device
    nz, nx = v.shape
    # replicate pad of width npml on all sides (use F.pad on 4-D)
    v4 = v.unsqueeze(0).unsqueeze(0)
    import torch.nn.functional as F
    v_pml = F.pad(v4, (npml, npml, npml, npml), mode="replicate")[0, 0]

    cmin = float(v.min())
    damp = abc_get_damp2d(nx, nz, npml, dx, cmin, device=device)

    dtdx = dt / dx
    alpha = (v_pml * dtdx) ** 2
    kappa = damp * dt
    c1t2 = 2.0 * C81
    temp1 = (2.0 + c1t2 * alpha - kappa).to(DTYPE)
    temp2 = (1.0 - kappa).to(DTYPE)
    beta_dt = (v_pml * dt) ** 2
    return alpha.to(DTYPE), temp1, temp2, beta_dt.to(DTYPE)


def _hanning(n: int, device=None) -> torch.Tensor:
    """Fortran ``hanning(n)``: 0.5 * (1 - cos(2π i / (2n+1))), i=1..n."""
    m = 2.0 * n + 1.0
    i = torch.arange(1, n + 1, dtype=DTYPE, device=device)
    return 0.5 * (1.0 - torch.cos(2.0 * math.pi * i / m))


def build_taper(nx: int, ntaper: int,
                device: torch.device | str | None = None) -> torch.Tensor:
    taper = torch.ones(nx, dtype=DTYPE, device=device)
    n2 = ntaper * 2
    h = _hanning(n2, device=device)
    taper[:n2] = h
    taper[nx - n2:] = h.flip(0)
    return taper
