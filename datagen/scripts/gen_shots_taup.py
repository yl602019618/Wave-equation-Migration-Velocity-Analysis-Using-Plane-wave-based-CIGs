"""Generate 201 point-source shot gathers with full-FD acoustic modeling
(IsFS=.true.), then tau-p slant-stack into 41 plane-wave CSGs that match the
geometry of the author's ``cpg_*.bin``.

This follows the paper's description:
  "Synthetic shot gathers are computed by finite-difference solutions to
  the 2D acoustic wave equation ... 201 shots with 402 active receivers."

Pipeline:
  (a) 201 point sources spread across x ∈ [0, 2000] m at 10 m spacing
  (b) per shot: record 402 receivers at x ∈ [0, 2005] m, 5 m spacing, at FS+1 depth
  (c) slant-stack: cpg(t, xg; p) = Σ_xs  sg(t + p*(xg - xs)/dt, xg, xs)
      for each of the 41 ray parameters from coord_pw_45t45.dat
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/home/pisquare/zhijun/pwmva_fortran/pwmva_torch')
from pwmva_torch.device import DTYPE, pick_device
from pwmva_torch.fd2d.pml import build_alpha_temp, build_taper
from pwmva_torch.fd2d.stencil import kernel_step

NZ, NX, DX = 201, 801, 2.5
NT, DT = 3000, 0.0006
NPML = 100
NZP, NXP = NZ + 2*NPML, NX + 2*NPML

PKG = Path('/home/pisquare/zhijun/pwmva_fortran/pwmva_package')


def read2d(p, nz, nx):
    return np.fromfile(p, '<f4').reshape(nx, nz).T

def read1d(p, n):
    return np.fromfile(p, '<f4')[:n]


def run_shots_batched(v_np, src_np, xs_grid, xg_grid, zp1, igz, device, batch=32):
    """Batched full-FD point-source shot modeling with FS.

    Returns seis: (NS, NT, NG) float32 numpy.
    """
    NS = len(xs_grid); NG = len(xg_grid)
    v = torch.from_numpy(v_np.astype(np.float32)).to(device)
    alpha, temp1, temp2, beta_dt = build_alpha_temp(v, DX, DT, NPML)
    s = torch.from_numpy(src_np.astype(np.float32)).to(device)

    xs_pml = torch.as_tensor([NPML + ix for ix in xs_grid], dtype=torch.long, device=device)
    xg_pml = torch.as_tensor([NPML + ix for ix in xg_grid], dtype=torch.long, device=device)
    zp1_t = int(zp1)
    igz_t = int(igz)

    seis_all = np.zeros((NS, NT, NG), dtype=np.float32)
    for b0 in range(0, NS, batch):
        B = min(batch, NS - b0)
        p0 = torch.zeros(B, NZP, NXP, device=device, dtype=DTYPE)
        p1 = torch.zeros_like(p0)
        p  = torch.zeros_like(p0)
        # batched shot index in PML
        b_idx = torch.arange(B, device=device)
        xs_b  = xs_pml[b0:b0+B]
        seis_b = torch.zeros(B, NT, NG, device=device, dtype=DTYPE)
        for it in range(1, NT+1):
            # kernel_step: in-place, writes p[:, iz0:iz1, ix0:ix1] from p0, p1
            kernel_step(p, p0, p1, alpha, temp1, temp2,
                        iz0=4, iz1=NZP-4, ix0=4, ix1=NXP-4)
            # Point-source injection at (zp1, xs)
            itt = it - 1
            if 0 <= itt < NT:
                # p[b, zp1, xs_b[b]] += beta_dt[zp1, xs_b[b]] * src[itt]
                bd = beta_dt[zp1_t, xs_b]                 # (B,)
                p[b_idx, zp1_t, xs_b] = p[b_idx, zp1_t, xs_b] + bd * s[itt]
            # Free surface: p[NPML, :] = 0; antisymm ghost
            p[:, NPML, :] = 0.0
            for iz in range(1, 5):
                p[:, NPML-iz, :] = -p[:, NPML+iz, :]
            # store at receivers
            seis_b[:, it-1, :] = p[:, igz_t, xg_pml]
            # rotate
            p0, p1 = p1, p
            p = torch.zeros_like(p0)
        seis_all[b0:b0+B] = seis_b.cpu().numpy()
        print(f"  shots {b0+1}..{b0+B}/{NS}  done", flush=True)
    return seis_all


def tau_p_stack(shots, xs_m, xg_m, p_list, dt, v_surf=1500.0, mute_buf_it=170):
    """Plane-wave synthesis from common-shot gathers with per-shot direct-wave mute.

    Formula:  cpg(t, xg; p) = Σ_is  sg[is](t - shift_is, xg)
    where:
      shift_is = p * (xs_is - xref) / dt,
      xref = max(xs_m)  if p < 0 else min(xs_m)
             (physical start of plane-wave wavefront: rightmost for left-going,
              leftmost for right-going; matches Fortran ``xref_1`` convention
              when p<0, and mirrors it for p>0 so *both* sides see a clean
              plane-wave with no negative-time truncation).

    Direct wave in each shot gather is pre-muted:
        sg[is](t, xg) := 0  for t < |xg - xs_is| / v_surf + mute_buf_it*dt
    """
    NS, NT, NG = shots.shape
    NP = len(p_list)
    mute = shots.copy()
    for is_ in range(NS):
        for ig in range(NG):
            t_dir = abs(xg_m[ig] - xs_m[is_]) / v_surf
            cut = int(t_dir / dt) + mute_buf_it
            if cut < NT:
                mute[is_, :cut, ig] = 0.0
            else:
                mute[is_, :, ig] = 0.0
    cpg = np.zeros((NP, NT, NG), dtype=np.float32)
    xs_max, xs_min = float(xs_m.max()), float(xs_m.min())
    for ip, p in enumerate(p_list):
        xref = xs_max if p < 0 else xs_min
        for is_ in range(NS):
            shift = int(round(p * (xs_m[is_] - xref) / dt))
            if shift >= 0:
                if shift < NT:
                    cpg[ip, shift:, :] += mute[is_, :NT-shift, :]
            else:
                s = -shift
                if s < NT:
                    cpg[ip, :NT-s, :] += mute[is_, s:, :]
    return cpg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--nshots', type=int, default=201)
    ap.add_argument('--test', action='store_true', help='1-shot sanity test only')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    v_true = read2d(PKG/'model/2D_models/mlayer/vel_201x801x2.5m.bin', NZ, NX)
    src   = read1d(PKG/'results/2D_models/mlayer/fdcsg_f40/source.bin', NT)

    # Geometry (following coord_pw_45t45.dat: xref = 2000 m, receivers 0..2000 by 5 m)
    # Paper: 201 shots, 402 receivers. Put 201 sources across x ∈ [0, 2000] m.
    xs_m = np.linspace(0.0, 2000.0, args.nshots)    # shot x positions in m
    xg_m = np.arange(401, dtype=np.float32) * 5.0   # 0,5,..,2000 (401 matches cpg)
    xs_grid = np.round(xs_m / DX).astype(int)       # grid indices [0..NX-1]
    xg_grid = np.round(xg_m / DX).astype(int)
    zp1 = NPML + 1 + 1   # +1 for IsFS=.true. (zp1 shifted down)
    igz = NPML + 0 + 1 + 1  # zg=0 → NPML+1, +1 for FS

    if args.test:
        xs_grid = xs_grid[args.nshots // 2 : args.nshots // 2 + 1]
        xs_m = xs_m[args.nshots // 2 : args.nshots // 2 + 1]

    print(f"NS={len(xs_grid)}  NG={len(xg_grid)}  zp1={zp1}  igz={igz}  device={device}")
    t0 = time.time()
    shots = run_shots_batched(v_true, src, xs_grid, xg_grid, zp1, igz, device,
                              batch=args.batch)
    print(f"shot FD:   {time.time()-t0:.1f} s")
    np.save(args.out / 'shots.npy', shots)
    # also save geometry
    np.savez(args.out / 'geom.npz', xs_m=xs_m, xg_m=xg_m)

    if args.test:
        print(f"shot[0] max={np.abs(shots[0]).max():.3e}")
        return

    # Read ray parameters from coord file
    coord = np.loadtxt(PKG/'model/2D_models/mlayer/coord_pw_45t45.dat')
    # col 1=PW, col 4=p. Take one p per PW.
    p_per_pw = []
    for pw in range(1, 42):
        mask = coord[:, 0] == pw
        p_per_pw.append(coord[mask, 3][0])
    p_list = np.array(p_per_pw)
    print(f"p range: [{p_list.min():.4e}, {p_list.max():.4e}]")

    t0 = time.time()
    cpg = tau_p_stack(shots, xs_m, xg_m, p_list, DT)
    print(f"tau-p:     {time.time()-t0:.1f} s")

    # Save each PW as a separate file
    for ip in range(41):
        # write as (ng, nt) F-order float32
        cpg[ip].T.astype('<f4').tofile(args.out / f'cpg_taup_{ip+1}.bin')
    print(f"wrote 41 cpg_taup_*.bin to {args.out}")


if __name__ == '__main__':
    main()
