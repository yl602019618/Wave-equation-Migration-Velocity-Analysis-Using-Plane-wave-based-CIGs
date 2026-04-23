# Data generation — reproducing observed plane-wave CSGs from scratch

This subdirectory reproduces `results/observed/cpg/cpg_{1..41}.bin` (the 41
plane-wave common-shot gathers shipped with the repo as observed data) from
the TRUE velocity model alone. Everything here is a *blind* reconstruction:
no author-specific post-processing scripts are assumed; only the
pipeline implied by the paper plus 2D acoustic FD primitives.

## 1. Pipeline

Follows the paper's description literally:

> "Synthetic shot gathers are computed by finite-difference solutions to the
> 2D acoustic wave equation ... Ricker wavelet with a 40-Hz peak frequency,
> and there are 201 shots with 402 active receivers per shot."
> — Guo & Schuster 2017, §Numerical examples

Three stages in `scripts/gen_shots_taup.py`:

1. **201 point-source shot gathers** — full 2D acoustic FD, 8th-order space /
   2nd-order time, PML=100, `IsFS=.true.` (Dirichlet FS at the top,
   antisymmetric pressure ghost). Batched on GPU (~30 s on an RTX 5090).
   Geometry:
   - Sources at `xs ∈ [0, 2000] m`, 10 m spacing (201 shots)
   - Receivers at `xg ∈ [0, 2000] m`, 5 m spacing (401 recs — matches shipped
     `cpg_*.bin` shape; the paper says 402 but the shipped data is 401)
   - `NT=3000`, `DT=0.6 ms` → 1.8 s record
   - TRUE velocity `results/model/vel_true_201x801x2.5m.bin`, 40-Hz Ricker
     from `results/observed/source.bin`
2. **Per-shot direct-wave mute** — for every single shot gather, zero samples
   with `t < |xg − xs| / v_surface + 0.102 s` before stacking. `v_surface =
   1500 m/s` (surface velocity of the TRUE model); the 0.102 s buffer (170
   samples) lets the Ricker tail die out after direct arrival.
3. **Tau-p slant-stack** into 41 plane-wave CSGs using ray parameters
   from `coord_pw_45t45.dat`:
   ```
   cpg(t, xg; p) = Σ_xs  sg_muted[xs](t − p·(xs − xref_p)/dt, xg)
   xref_p = max(xs)  if p < 0
   xref_p = min(xs)  if p > 0
   ```
   The `xref_p` sign-flip is the non-obvious part: for `p > 0` the reference
   must sit at the **left** edge so no source has a negative effective
   fire-time; mirror for `p < 0`. Using a single `xref_1 = 2000` (what the
   Fortran parfile defaults to) only works for one half of the ray parameters.

The pipeline is data-only — no calls into the Fortran `pwmva_package`. It
reads only the TRUE velocity, the Ricker source wavelet, and the ray-parameter
list.

## 2. Results vs shipped cpg

**Data match** (per-shot Pearson correlation with `results/observed/cpg/cpg_{k}.bin`):

| metric | value |
|--------|-------|
| mean (41 shots) | **0.9619** |
| min (edge shots 1, 41) | 0.8186 |
| max (mid shots 11, 31) | 0.9768 |

Symmetric by design: `Pearson[k] == Pearson[42-k]`. Edge-shot cap at ~0.82
is aperture-limited; see §4.

Visual comparison — see `viz/taup_final_shot{11,21,31}.png` (three-panel:
author ref / ours scaled / residual). Residual is mostly band-limited
numerical noise.

**20-iter inversion** running the torch PWEMVA with `--cpg-dir` pointing at
the generated data, compared to the inversion on the shipped `cpg_*.bin`:

| iter | our data → TRUE | shipped cpg → TRUE | Δ (ours − shipped) |
|------|-----------------|---------------------|---------------------|
| init velh | 155.902 m/s | 155.902 | — |
| 1 | 146.755 | 146.721 | +0.03 |
| 5 | **139.241** | 139.810 | **−0.57** |
| 10 | **136.171** | 138.878 | **−2.71** |
| 15 | **135.993** | 138.983 | **−2.99** |
| 20 | 136.115 | 137.466 | **−1.35** |

Our generated data actually converges **faster** than the shipped cpg in
iters 5-15 (2-3 m/s closer to TRUE). Misfit flattens around iter 10;
iters 11-13 hit the numerical floor (line-search 7× backtracks, fff=7.6e-8)
and recover by iter 14. Iter 20 regresses slightly to 136.1 due to
late-iteration line-search oscillation — at that point the 4% Pearson gap
in the data behaves like floor-level noise.

See figures:
- `viz/v1_convergence_20iter.png` — RMS vs TRUE + misfit curve across 20 iters
- `viz/v1_evolution_20iter.png` — velinv snapshots at iters 1/5/10/15/20
- `viz/v1_iter20_vs_author.png` — side-by-side our vs shipped iter 20, plus
  diff map (max |Δ| ≈ 103 m/s, localized at anomaly edges)

## 3. Contents

```
datagen/
├── README.md                                     this file
├── metrics.json                                  frozen per-shot Pearson + per-iter RMS
├── scripts/gen_shots_taup.py                     full pipeline (single file)
├── inversion/velinv_{1,5,10,15,20}.bin           reconstructed velocity snapshots
├── inversion/misfits_20iter.npy                  misfit_tot per iter
├── inversion/fffs_20iter.npy                     line-search fff per iter
└── viz/
    ├── taup_final_shot{11,21,31}.png             data-match (ref / ours*scale / residual)
    ├── v1_convergence_20iter.png                 RMS + misfit convergence
    ├── v1_evolution_20iter.png                   velinv snapshots across iters
    ├── v1_iter20_vs_author.png                   our vs shipped iter 20 + diff map
    └── inversion_taup_vs_scattered_vs_author.png earlier 3×3 iter-1/5 comparison
```

## 4. Known residual differences

- **Edge shots (1, 41) cap at Pearson 0.82** — aperture-limited: at ±45°
  the plane wave takes 1/cos(45°) ≈ 1.4× longer path through PML, and the
  201-shot range `[0, 2000] m` is not wide enough to fully form the plane
  wave at the maximum tilt. Even with a perfect oracle tail-mask the cap
  is 0.847, meaning the gap is **inside the signal window**, not a tail
  artifact. Fixing this likely needs the Fortran `a2d_pw1_mod` FD directly
  (plane-wave source injection at every grid point = infinitely fine tau-p).
- **Mid-shot cap at Pearson 0.977** — residual ~2% is probably from
  subtle FS / stencil / PML conventions that differ between the shipped
  implementation and our torch port.

v2 attempts — linear interpolation for fractional shifts, shorter mute
buffer, more shots (401/801 at 5 m / 2.5 m spacing), NPML=200, different
`xref` conventions, ref-envelope post-mute — were all tested systematically.
None produced a net improvement: when Pearson rose marginally (0.962 → 0.969
with a shorter mute), the late-iter line search became unstable (fff
collapsed at iter 4 and inversion RMS regressed). The v1 defaults above
strike the best balance.

## 5. Reproduce

```bash
# Regenerate the 41 plane-wave CSGs (~30 s on RTX 5090):
python datagen/scripts/gen_shots_taup.py \
    --out /tmp/cpg_taup --nshots 201 --batch 32

# Run 20-iter torch inversion on the generated data:
cd pwmva_torch
python scripts/run_iter.py --niter 20 --batch 8 --no-3d-dump \
    --out /tmp/torch_taup_iter20 \
    --cpg-dir /tmp/cpg_taup --cpg-pat 'cpg_taup_{i}.bin'
```

The script only needs the paths in the paper's package structure; edit the
`PKG` constant at the top of `gen_shots_taup.py` if your `pwmva_package` is
elsewhere.
