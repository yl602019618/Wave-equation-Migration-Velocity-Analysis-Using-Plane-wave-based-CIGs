"""T4 parity: batched torch RTM vs Fortran image_tot_1.bin and numpy pw1_rtm.

Pipeline (matches ``inversion._rtm_worker`` in the numpy port):

  img = pw1_rtm(velh, cpg_is) → image_col = -z_derivative_2d(img)

We check the entire 41-PW volume against ``image_tot_1.bin``, and a couple of
individual PWs with tighter Pearson thresholds.
"""
from pathlib import Path
import time
import numpy as np
import torch

from pwmva_torch.device import pick_device
from pwmva_torch.source import ricker
from pwmva_torch.fd2d.pw1_rtm import pw1_rtm_batch
from pwmva_torch.image_ops import z_derivative_2d

# numpy oracle
from pwmva.io import read2d as read2d_np, read3d as read3d_np
from pwmva.config import read_parfile
from pwmva.geom  import read_pw1_coord

from _viz import plot_2d_compare, pearson, rms_rel

DEVICE = pick_device()
NZ, NX, NS = 201, 801, 41


def _gather_pw_batch(coord, device):
    ng = int(coord.ng[0])
    p_ray = torch.from_numpy(coord.p.astype(np.float32)).to(device)
    xref  = torch.from_numpy(coord.xref_1.astype(np.float32)).to(device)
    zp1   = torch.from_numpy(coord.zp1.astype(np.float32)).to(device)
    xg = torch.from_numpy(coord.xg[:ng, :].T.astype(np.float32)).to(device)   # (ns, ng)
    zg = torch.from_numpy(coord.zg[:ng, :].T.astype(np.float32)).to(device)
    return p_ray, xref, zp1, xg, zg, ng


def _run_rtm_all(v_np, par, coord, cpgs, device, batch: int = 41):
    """Run RTM for all 41 PWs, possibly in sub-batches, and assemble image_tot."""
    src = ricker(par.nw, par.dt, par.freq, device=device)
    v = torch.from_numpy(v_np).to(device)
    p_ray, xref, zp1, xg, zg, ng = _gather_pw_batch(coord, device)
    seis = torch.from_numpy(cpgs.astype(np.float32)).to(device)    # (ns, nt, ng)

    image_tot = torch.zeros((NS, NZ, NX), dtype=torch.float32, device=device)
    for b0 in range(0, NS, batch):
        b1 = min(NS, b0 + batch)
        res = pw1_rtm_batch(
            v, par.dx, par.dt, par.nt, par.npml, src,
            p_ray[b0:b1], xref[b0:b1], zp1[b0:b1],
            xg[b0:b1, :], zg[b0:b1, :],
            seis[b0:b1], xs_taper_grid=40, dt_record=6,
        )
        # image_col = -z_derivative_2d(img, fd_order=4)
        image_tot[b0:b1] = -z_derivative_2d(res.img, fd_order=4)
    return image_tot


def test_rtm_image_tot_1(model_dir, csg_dir, working_dir, rerun_dir, viz_dir):
    par = read_parfile(working_dir / "parfile_pwmva_warp_rerun.sh")
    coord = read_pw1_coord(model_dir / "coord_pw_45t45.dat")
    v_np = read2d_np(model_dir / "velh_201x801x2.5m.bin", NZ, NX).astype(np.float32)

    # load all 41 observed shot gathers
    ng = int(coord.ng[0])
    cpgs = np.stack([
        read2d_np(csg_dir / "cpg" / f"cpg_{i+1}.bin", par.nt, ng).astype(np.float32)
        for i in range(NS)
    ])                                                      # (ns, nt, ng)

    t0 = time.time()
    image_tot_t = _run_rtm_all(v_np, par, coord, cpgs, DEVICE, batch=8)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    image_tot_np = image_tot_t.detach().to("cpu").numpy().transpose(1, 2, 0)   # (nz, nx, ns)

    # Fortran oracle
    image_tot_ref = read3d_np(rerun_dir / "image_tot_1.bin", NZ, NX, NS).astype(np.float32)

    p_full = pearson(image_tot_ref, image_tot_np)
    r_full = rms_rel(image_tot_ref, image_tot_np)
    peak_gpu = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[T4] 41 PW RTM on GPU: {elapsed:.2f} s  peak GPU mem ~ {peak_gpu:.2f} GB")
    print(f"     image_tot_1.bin full-volume: pearson={p_full:.6f}  rms_rel={r_full:.3e}")
    assert p_full > 0.9999

    # Per-PW checks: 21 (pilot), 11, 31
    for is_ in [20, 10, 30]:
        a = image_tot_ref[:, :, is_]
        b = image_tot_np[:, :, is_]
        p = pearson(a, b)
        r = rms_rel(a, b)
        print(f"     is={is_+1}: pearson={p:.6f}  rms_rel={r:.3e}")
        assert p > 0.9999
        plot_2d_compare(a, b,
                        [f"Fortran image_tot is={is_+1}",
                         f"torch image_tot is={is_+1}", "diff"],
                        viz_dir / f"t4_image_tot_is{is_+1}.png",
                        clip_percentile=99.5)
