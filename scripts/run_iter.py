"""Run one or more PWEMVA iterations on GPU and dump intermediates.

Usage:
  python scripts/run_iter.py --niter 1 --batch 8 --out /tmp/torch_iter1
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch

from pwmva_torch import io as io_t
from pwmva_torch.device import pick_device
from pwmva_torch.inversion import IterState, build_context, run_iteration

# numpy helpers to parse parfile / geometry (same file format as Fortran)
from pwmva.config import read_parfile
from pwmva.geom   import read_pw1_coord
from pwmva.io     import read1d as read1d_np, read2d as read2d_np


PKG = Path("/home/pisquare/zhijun/pwmva_fortran/pwmva_package")
NZ, NX, NS = 201, 801, 41


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niter", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8,
                    help="plane-wave batch size per RTM/wavepath call")
    ap.add_argument("--device", type=str, default=None, help="e.g. cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume-from", type=int, default=0,
                    help="resume from velinv_<N>.bin in --out (start iter = N+1)")
    ap.add_argument("--no-3d-dump", action="store_true",
                    help="skip image_tot_<N>.bin and shift_<N>.bin")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    par = read_parfile(PKG / "working" / "parfile_pwmva_warp_rerun.sh")
    coord = read_pw1_coord(PKG / "model" / "2D_models" / "mlayer" / "coord_pw_45t45.dat")
    src_np = read1d_np(PKG / "results/2D_models/mlayer/fdcsg_f40/source.bin", par.nw)
    ng = int(coord.ng[0])
    cpgs = np.stack([
        read2d_np(PKG / f"results/2D_models/mlayer/fdcsg_f40/cpg/cpg_{i+1}.bin",
                  par.nt, ng).astype(np.float32)
        for i in range(NS)
    ])

    ctx = build_context(par, coord, cpgs, src_np, device=device, batch=args.batch)

    if args.resume_from > 0:
        v0_np = read2d_np(args.out / f"velinv_{args.resume_from}.bin", NZ, NX).astype(np.float32)
        start = args.resume_from + 1
        print(f"resumed from velinv_{args.resume_from}.bin → start iter {start}")
    else:
        v0_np = read2d_np(PKG / "model/2D_models/mlayer/velh_201x801x2.5m.bin",
                          NZ, NX).astype(np.float32)
        start = 1
    v0 = torch.from_numpy(v0_np).to(device)
    print(f"device={device} batch={args.batch} NS={NS} nt={par.nt}")

    state = IterState(v=v0, iter_idx=start)
    misfits = []
    fffs = []
    t_total = time.time()
    for k in range(args.niter):
        t0 = time.time()
        state = run_iteration(state, ctx, debug_dir=args.out,
                              skip_3d_dump=args.no_3d_dump)
        print(f"iter {state.iter_idx - 1} done in {time.time() - t0:.1f}s  "
              f"misfit={state.misfit:.3e}  fff={state.fff:.3e}", flush=True)
        misfits.append(state.misfit); fffs.append(state.fff)
    print(f"total {args.niter} iters in {time.time() - t_total:.1f}s")
    np.save(args.out / f"misfits_resume{args.resume_from}.npy",
            np.array(misfits))
    np.save(args.out / f"fffs_resume{args.resume_from}.npy",
            np.array(fffs))


if __name__ == "__main__":
    main()
