"""Batched plane-wave acoustic RTM — port of `pwmva_python.fd2d.pw1_rtm`.

Pipeline per batch of B plane waves (all sharing velocity v):
  1. Forward-propagate source wavefield; subsample to ``wfs[B, nz, nx, ntr]``.
  2. Back-propagate receiver wavefield injecting observed ``seis``; subsample
     to ``wfr[B, nz, nx, ntr]``.
  3. Image: ``img[b] = Σ_t wfs[b,:,:,t] * wfr[b,:,:,t]``.

Fortran defaults: ``xs_taper_grid=40`` (half the modeling value), ``dt_record=6``.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from ..device import DTYPE
from .pml import build_alpha_temp, build_taper
from .stencil import kernel_step


@dataclass
class RTMBatchResult:
    img: torch.Tensor    # (B, nz, nx)


@torch.no_grad()
def pw1_rtm_batch(
    v: torch.Tensor,             # (nz, nx)
    dx: float, dt: float, nt: int, npml: int,
    src: torch.Tensor,           # (nw,)
    p_ray:    torch.Tensor,      # (B,)
    xref_1:   torch.Tensor,      # (B,)
    zp1_phys: torch.Tensor,      # (B,)
    xg: torch.Tensor,            # (B, ng)
    zg: torch.Tensor,            # (B, ng)
    seis: torch.Tensor,          # (B, nt, ng) observed data (cpg)
    xs_taper_grid: int = 40,
    dt_record: int = 6,
) -> RTMBatchResult:
    device = v.device
    B = p_ray.shape[0]
    nz, nx = v.shape
    nz_pml = nz + 2 * npml
    nx_pml = nx + 2 * npml

    alpha, temp1, temp2, beta_dt = build_alpha_temp(v, dx, dt, npml)
    taper = build_taper(nx, xs_taper_grid, device=device)

    zp1_scalar = int(torch.round(zp1_phys[0] / dx).item()) + npml
    # per-PW grid indices
    igx = (npml + (xg / dx).to(torch.int64))
    igz = (npml + (zg / dx).to(torch.int64))
    ng = xg.shape[1]

    # delays (B, nx)
    i_phys = torch.arange(nx, dtype=torch.float64, device=device)
    delays = torch.round(p_ray.double().unsqueeze(1)
                         * (i_phys.unsqueeze(0) * dx - xref_1.double().unsqueeze(1))
                         / dt).to(torch.int64)

    ix_pml = torch.arange(npml, npml + nx, dtype=torch.int64, device=device)
    bd_row = beta_dt[zp1_scalar, ix_pml]              # (nx,)
    # receiver-side β*dt² (B, ng)
    bd_recv = beta_dt[igz, igx]

    src = src.to(device=device, dtype=DTYPE)
    seis = seis.to(device=device, dtype=DTYPE)
    nw = src.shape[0]

    iz0, iz1 = 4, nz_pml - 4
    ix0, ix1 = 4, nx_pml - 4

    nt_record = int(round(nt / dt_record)) + 1
    wfs = torch.zeros((B, nz, nx, nt_record), dtype=DTYPE, device=device)
    wfr = torch.zeros((B, nz, nx, nt_record), dtype=DTYPE, device=device)

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, ng)  # (B, ng)
    b_src     = torch.arange(B, device=device).unsqueeze(1).expand(-1, nx)
    ix_src    = ix_pml.unsqueeze(0).expand(B, -1)

    # ---------- 1. Forward source wavefield ----------
    p0 = torch.zeros((B, nz_pml, nx_pml), dtype=DTYPE, device=device)
    p1 = torch.zeros_like(p0)
    p  = torch.zeros_like(p0)

    for it in range(1, nt + 1):
        kernel_step(p, p0, p1, alpha, temp1, temp2, iz0, iz1, ix0, ix1)

        itt = it - delays
        valid = (itt >= 1) & (itt <= nw)
        idx_t = torch.clamp(itt - 1, 0, nw - 1)
        src_val = torch.where(valid, src[idx_t], torch.zeros_like(src[idx_t]))
        add = (bd_row.unsqueeze(0) * src_val * taper.unsqueeze(0))
        p[b_src, zp1_scalar, ix_src] = p[b_src, zp1_scalar, ix_src] + add

        it_rec = int(round(it / dt_record))
        wfs[:, :, :, it_rec] = p[:, npml:npml + nz, npml:npml + nx]

        p0, p1, p = p1, p, p0

    # ---------- 2. Back-propagate receiver wavefield ----------
    p0.zero_(); p1.zero_(); p.zero_()

    for it in range(nt, 0, -1):
        kernel_step(p, p0, p1, alpha, temp1, temp2, iz0, iz1, ix0, ix1)
        # inject observed data at receivers: p[b, igz[b,g], igx[b,g]] += bd_recv[b,g] * seis[b, it-1, g]
        add = bd_recv * seis[:, it - 1, :]                     # (B, ng)
        p[batch_idx, igz, igx] = p[batch_idx, igz, igx] + add

        it_rec = int(round(it / dt_record))
        wfr[:, :, :, it_rec] = p[:, npml:npml + nz, npml:npml + nx]

        p0, p1, p = p1, p, p0

    # ---------- 3. Image condition: Σ_t wfs * wfr ----------
    img = (wfs * wfr).sum(dim=-1)                              # (B, nz, nx)
    return RTMBatchResult(img=img)
