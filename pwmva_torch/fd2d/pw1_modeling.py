"""Batched plane-wave acoustic forward modeling.

Port of ``pwmva_python.fd2d.pw1_modeling`` but every operation runs
simultaneously over a ``B``-axis of plane waves.

Each plane wave has its own ``p_ray``, ``xref_1`` and receiver coordinates
(``xg``, ``zg``), but all share the same velocity ``v`` so the PML
coefficients are shared across the batch.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from ..device import DTYPE
from .pml import build_alpha_temp, build_taper
from .stencil import kernel_step


@dataclass
class PW1ModelingBatchResult:
    seis: torch.Tensor     # (B, nt, ng) at receiver positions
    snapshots: dict | None # optional {it: p_phys (B, nz, nx)}


@torch.no_grad()
def pw1_modeling_batch(
    v: torch.Tensor,             # (nz, nx) float32 m/s
    dx: float,
    dt: float,
    nt: int,
    npml: int,
    src: torch.Tensor,           # (nw,)
    p_ray:   torch.Tensor,       # (B,) float  plane-wave ray parameters
    xref_1:  torch.Tensor,       # (B,) float  reference x per PW
    zp1_phys: torch.Tensor,      # (B,) float  source depth (m) per PW
    xg: torch.Tensor,            # (B, ng) float
    zg: torch.Tensor,            # (B, ng) float
    xs_taper_grid: int = 80,
    snapshot_every: int | None = None,
) -> PW1ModelingBatchResult:
    """Forward-model ``B`` plane waves at once.

    All per-PW arrays must be on the same device. Receiver count ``ng`` is
    assumed uniform across the batch (in ``mlayer`` all PWs use 401 receivers).
    """
    device = v.device
    B = p_ray.shape[0]
    nz, nx = v.shape
    nz_pml = nz + 2 * npml
    nx_pml = nx + 2 * npml

    alpha, temp1, temp2, beta_dt = build_alpha_temp(v, dx, dt, npml)
    taper = build_taper(nx, xs_taper_grid, device=device)                 # (nx,)

    # All PWs share zp1_phys = 0 in mlayer, but still compute per-PW for generality.
    zp1 = (torch.round(zp1_phys / dx).to(torch.int64) + npml)              # (B,)
    zp1_scalar = int(zp1[0].item())
    assert bool((zp1 == zp1_scalar).all()), "zp1 must be uniform across batch"

    # Receiver grid indices (B, ng)
    igx = (npml + (xg / dx).to(torch.int64))
    igz = (npml + (zg / dx).to(torch.int64))
    ng  = xg.shape[1]

    # Delays per (B, nx): nint(p * (ix*dx - xref_1) / dt)
    i_phys = torch.arange(nx, dtype=torch.float64, device=device)          # (nx,)
    delays = torch.round(p_ray.double().unsqueeze(1)
                         * (i_phys.unsqueeze(0) * dx - xref_1.double().unsqueeze(1))
                         / dt).to(torch.int64)                             # (B, nx)

    ix_pml = torch.arange(npml, npml + nx, dtype=torch.int64, device=device)
    # bd_row at zp1 across the batch — shape (B, nx). Since zp1 is uniform, use scalar row.
    bd_row = beta_dt[zp1_scalar, ix_pml]                                   # (nx,)
    src = src.to(dtype=DTYPE, device=device)
    nw = src.shape[0]

    p0 = torch.zeros((B, nz_pml, nx_pml), dtype=DTYPE, device=device)
    p1 = torch.zeros_like(p0)
    p  = torch.zeros_like(p0)
    seis = torch.zeros((B, nt, ng), dtype=DTYPE, device=device)
    snapshots = {} if snapshot_every else None

    # 8th-order interior region (5..nz_pml-4 in 1-based → 4..nz_pml-5 in 0-based)
    iz0, iz1 = 4, nz_pml - 4
    ix0, ix1 = 4, nx_pml - 4

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, ng)  # (B, ng)
    b_src     = torch.arange(B, device=device).unsqueeze(1).expand(-1, nx)  # (B, nx)
    ix_src    = ix_pml.unsqueeze(0).expand(B, -1)                           # (B, nx)

    for it in range(1, nt + 1):
        kernel_step(p, p0, p1, alpha, temp1, temp2, iz0, iz1, ix0, ix1)

        # Plane-wave source injection at row zp1, all physical ix.
        # itt = it - delays  (B, nx);  valid where 1 ≤ itt ≤ nw.
        itt = it - delays
        valid = (itt >= 1) & (itt <= nw)
        idx = torch.clamp(itt - 1, 0, nw - 1)
        src_val = torch.where(valid, src[idx], torch.zeros_like(src[idx]))
        add = (bd_row.unsqueeze(0) * src_val * taper.unsqueeze(0))          # (B, nx)
        # advanced-indexing scatter along (B, zp1, ix_pml)
        p[b_src, zp1_scalar, ix_src] = p[b_src, zp1_scalar, ix_src] + add

        # Record receivers: p[b, igz[b,g], igx[b,g]]  → (B, ng)
        seis[:, it - 1, :] = p[batch_idx, igz, igx]

        if snapshot_every and (it % snapshot_every == 1 or it == nt):
            snapshots[it] = p[:, npml:npml + nz, npml:npml + nx].clone()

        # rotate buffers (p0 ← p1 ← p)
        p0, p1, p = p1, p, p0

    return PW1ModelingBatchResult(seis=seis, snapshots=snapshots)
