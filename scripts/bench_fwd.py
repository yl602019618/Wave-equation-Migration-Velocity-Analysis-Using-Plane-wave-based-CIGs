"""Benchmark batched FWD modeling on GPU. Also writes a per-PW shotgather
visualisation (21/11/31) alongside Fortran oracles."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch

from pwmva_torch.device import pick_device
from pwmva_torch.source import ricker
from pwmva_torch.fd2d.pw1_modeling import pw1_modeling_batch

from pwmva.io import read2d as read2d_np
from pwmva.config import read_parfile
from pwmva.geom  import read_pw1_coord

PKG = Path("/home/pisquare/zhijun/pwmva_fortran/pwmva_package")
DEVICE = pick_device()


def main():
    par = read_parfile(PKG / "working/parfile_pwmva_warp_rerun.sh")
    coord = read_pw1_coord(PKG / "model/2D_models/mlayer/coord_pw_45t45.dat")
    v_np = read2d_np(PKG / "model/2D_models/mlayer/vel_201x801x2.5m.bin", par.nz, par.nx).astype(np.float32)

    v = torch.from_numpy(v_np).to(DEVICE)
    src = ricker(par.nw, par.dt, par.freq, device=DEVICE)
    print(f"device={DEVICE} NT={par.nt} NZ={par.nz} NX={par.nx} NPML={par.npml} B=41")

    ng = int(coord.ng[0])
    p_ray = torch.from_numpy(coord.p.astype(np.float32)).to(DEVICE)
    xref  = torch.from_numpy(coord.xref_1.astype(np.float32)).to(DEVICE)
    zp1   = torch.from_numpy(coord.zp1.astype(np.float32)).to(DEVICE)
    xg = torch.from_numpy(coord.xg[:ng, :].T.astype(np.float32)).to(DEVICE)  # (41, ng)
    zg = torch.from_numpy(coord.zg[:ng, :].T.astype(np.float32)).to(DEVICE)

    # warm-up
    _ = pw1_modeling_batch(v, par.dx, par.dt, 10, par.npml, src,
                           p_ray, xref, zp1, xg, zg, xs_taper_grid=80)
    torch.cuda.synchronize()

    # timed
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = pw1_modeling_batch(v, par.dx, par.dt, par.nt, par.npml, src,
                             p_ray, xref, zp1, xg, zg, xs_taper_grid=80)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"41 PW × {par.nt} steps on 1 GPU: {elapsed:.2f} s   peak GPU mem = {peak:.2f} GB")
    print(f"per-step per-PW = {1000*elapsed/(par.nt * 41):.3f} ms")


if __name__ == "__main__":
    main()
