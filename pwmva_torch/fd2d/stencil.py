"""8th-order spatial / 2nd-order time FD stencil on a batched wavefield.

Wavefield tensors have shape ``(B, nz_pml, nx_pml)``; coefficient tensors
``alpha/temp1/temp2/beta_dt`` are shared across the batch — shape
``(nz_pml, nx_pml)`` — and broadcast implicitly.

Update (Fortran a2d.f90, 8th-order spatial stencil):

  p[n+1] = temp1*p[n] - temp2*p[n-1]
         + alpha * Σ_k C8_{k+1} * ( p[n, iz±k, ix] + p[n, iz, ix±k] )

where the k=0 diagonal (C81) contribution is folded into temp1 (c1t2 = 2·C81).
"""
from __future__ import annotations
import torch

from .pml import C82, C83, C84, C85


def kernel_step(p: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor,
                alpha: torch.Tensor, temp1: torch.Tensor, temp2: torch.Tensor,
                iz0: int, iz1: int, ix0: int, ix1: int) -> None:
    """In-place: ``p[:, iz0:iz1, ix0:ix1]`` ← next timestep from p0/p1.

    All inputs share ``(B, nz_pml, nx_pml)`` for the wavefields;
    ``alpha/temp1/temp2`` are ``(nz_pml, nx_pml)`` and broadcast.
    """
    c = p1[:, iz0:iz1, ix0:ix1]
    lap = (
        C82 * (p1[:, iz0:iz1,        ix0 + 1:ix1 + 1] + p1[:, iz0:iz1,        ix0 - 1:ix1 - 1]
             + p1[:, iz0 + 1:iz1 + 1, ix0:ix1]         + p1[:, iz0 - 1:iz1 - 1, ix0:ix1])
      + C83 * (p1[:, iz0:iz1,        ix0 + 2:ix1 + 2] + p1[:, iz0:iz1,        ix0 - 2:ix1 - 2]
             + p1[:, iz0 + 2:iz1 + 2, ix0:ix1]         + p1[:, iz0 - 2:iz1 - 2, ix0:ix1])
      + C84 * (p1[:, iz0:iz1,        ix0 + 3:ix1 + 3] + p1[:, iz0:iz1,        ix0 - 3:ix1 - 3]
             + p1[:, iz0 + 3:iz1 + 3, ix0:ix1]         + p1[:, iz0 - 3:iz1 - 3, ix0:ix1])
      + C85 * (p1[:, iz0:iz1,        ix0 + 4:ix1 + 4] + p1[:, iz0:iz1,        ix0 - 4:ix1 - 4]
             + p1[:, iz0 + 4:iz1 + 4, ix0:ix1]         + p1[:, iz0 - 4:iz1 - 4, ix0:ix1])
    )
    p[:, iz0:iz1, ix0:ix1] = (
        temp1[iz0:iz1, ix0:ix1] * c
        - temp2[iz0:iz1, ix0:ix1] * p0[:, iz0:iz1, ix0:ix1]
        + alpha[iz0:iz1, ix0:ix1] * lap
    )


def pertubation_ic2(p1: torch.Tensor, refl_pml: torch.Tensor,
                    beta_dt: torch.Tensor, q: torch.Tensor) -> None:
    """Fortran ``pertubation(ic=2)``: 5-pt Laplacian of p1 × refl × beta_dt,
    accumulated into q. Updates interior ``[:, 1:-1, 1:-1]``.

    - ``p1`` : (B, nz_pml, nx_pml)
    - ``refl_pml`` : (B, nz_pml, nx_pml)  — per-PW virtual reflectivity on PML grid
    - ``beta_dt`` : (nz_pml, nx_pml) broadcast
    - ``q`` : (B, nz_pml, nx_pml) accumulator (in-place add)
    """
    lap = (p1[:, :-2, 1:-1] + p1[:, 2:, 1:-1]
           + p1[:, 1:-1, :-2] + p1[:, 1:-1, 2:]
           - 4.0 * p1[:, 1:-1, 1:-1])
    q[:, 1:-1, 1:-1] = q[:, 1:-1, 1:-1] + lap * refl_pml[:, 1:-1, 1:-1] * beta_dt[1:-1, 1:-1]
