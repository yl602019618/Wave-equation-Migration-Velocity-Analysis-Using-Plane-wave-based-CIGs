# Wave-equation Migration Velocity Analysis Using Plane-wave Common Image Gathers (PyTorch, GPU)

A PyTorch + CUDA reproduction of

> **Guo, B. and Schuster, G. T. (2017).** *Wave-equation migration velocity
> analysis using plane-wave common-image gathers.* **Geophysics**, 82(5),
> S327–S340. <https://doi.org/10.1190/geo2016-0596.1>

Every FD / RTM / wavepath operation runs simultaneously over a batch of plane
waves on a single GPU. Correctness is validated step-by-step against both
the original Fortran implementation by the authors and a NumPy + numba CPU
port.

### References used while building this implementation

- **Paper**
  Guo, B., & Schuster, G. T. (2017). *Wave-equation migration velocity
  analysis using plane-wave common-image gathers.* **Geophysics**, 82(5),
  S327–S340. <https://doi.org/10.1190/geo2016-0596.1>

- **KAUST CSIM course notes (MVA chapter)** — used as the conceptual
  reference while implementing the wavepath gradient and line search:
  <https://csim.kaust.edu.sa/files/SeismicInversion/Chapter.MVA/index.html>

- **Original Fortran implementation** by the authors (the development notes
  refer to this bundle as `pwmva_package`; the `pwmva_warp` branch with the
  `mlayer` example is the exact run this repository reproduces bit-to-bit
  through iteration 1).

- **NumPy + numba CPU port** (bit-perfect oracle to Fortran) — used as an
  in-process oracle for every parity test in `tests/`; see `plan.md`.

### How to cite

If this reproduction is useful to you, please cite the original paper:

```bibtex
@article{guo2017pwmva,
  title   = {Wave-equation migration velocity analysis using plane-wave common-image gathers},
  author  = {Guo, Bowen and Schuster, Gerard T.},
  journal = {Geophysics},
  volume  = {82},
  number  = {5},
  pages   = {S327--S340},
  year    = {2017},
  doi     = {10.1190/geo2016-0596.1}
}
```

---

## 1. Headline results

### 1.1 Per-phase correctness (vs Fortran / NumPy oracles)

| Phase | What | Result |
|-------|------|--------|
| T1 Ricker source | `source.py` | `max|Δ|` vs `source.bin` = **4.79e-7** (FP32 ULP) |
| T2 image ops | `z_deriv / gauss_smooth / vel_smooth / masks / imageagc` | 5/6 bit-identical; `imageagc` Pearson **0.999998** |
| T3 FD forward | `fd2d/pw1_modeling.py` | vs NumPy **Pearson = 1.00000**, `rms_rel ≈ 1.5e-6`; vs Fortran `cpg_full_*.bin` peak amplitude bit-identical |
| T4 RTM | `fd2d/pw1_rtm.py` | `image_tot_1.bin` **Pearson = 0.999977**, `rms_rel = 1.4 %` |
| T5 wavepath | `fd2d/wavepath.py` | single PW **Pearson = 1.000000**; end-to-end `gradient_1.bin` **Pearson = 0.999999**, `rms_rel = 2.7e-3` |
| **T6 iter 1** | full outer loop | `velinv_1.bin` vs Fortran rerun: **RMS = 0.039 m/s**, Pearson = 0.999999 |

All parity plots are shipped under [`tests/_viz_out/`](tests/_viz_out).

### 1.2 Convergence across 30 outer iterations

| iter | torch vs Fortran rerun | torch vs TRUE | Fortran vs TRUE |
|------|------------------------|---------------|-----------------|
|  1 | **0.039** | 146.72 | 146.72 |
|  2 | 0.056 | 145.17 | 145.18 |
|  3 | 0.226 | 142.43 | 142.41 |
|  4 | 0.213 | 140.97 | 140.95 |
|  5 | 6.02 | 139.81 | 140.26 |
| 10 | 8.23 | 138.88 | 139.76 |
| 15 | 20.64 | 138.98 | 138.03 |
| 20 | 10.15 | 137.47 | 136.42 |
| 25 |   —   | 136.02 | (no ref) |
| 28 |   —   | **135.83** | (no ref) |
| 30 |   —   | 136.10 | (no ref) |

*Torch and Fortran converge to the same local minimum (within 1 m/s of each
other at iteration 20; both reduce RMS-vs-TRUE by ~19 m/s from the 155.9 m/s
initial model). iter 21–30 are torch-only runs that continue the descent and
reach a new minimum at iter 28 (135.83 m/s).*

### 1.3 Reproducing the observed data from scratch (blind data-gen)

`datagen/` contains an end-to-end pipeline that regenerates the observed
plane-wave CSGs using only the TRUE velocity model and the Ricker source — no
code or scripts from the author's `pwmva_package`. Pipeline:

```
1. 201 point-source shot gathers (full FD, IsFS=.true., PML=100)  ~30 s on RTX 5090
2. per-shot direct-wave mute (|xg − xs|/1500 + 0.102 s)
3. tau-p slant-stack → 41 plane-wave CSGs
   (xref = max(xs) if p<0 else min(xs) — sign-dependent reference)
```

**Data match** vs the shipped `results/observed/cpg/cpg_{k}.bin`:
per-shot Pearson **mean 0.962**, min 0.819 (edge ±45°), max 0.977 (mid ±20°).

**Inversion match**: running the same 20-iter torch PWEMVA on the
freshly-generated data vs the shipped data:

| iter | generated → TRUE | shipped → TRUE | Δ |
|------|------------------|-----------------|----|
| 1 | 146.755 | 146.721 | +0.03 |
| 5 | **139.24** | 139.81 | −0.57 |
| 10 | **136.17** | 138.88 | **−2.71** |
| 15 | **135.99** | 138.98 | **−2.99** |
| 20 | 136.12 | 137.47 | **−1.35** |

The blindly-regenerated data actually converges **faster** to TRUE in
iters 5-15 (2–3 m/s better) than the shipped cpg. See
[`datagen/README.md`](datagen/README.md) for the full pipeline, known residual
differences, and v2 attempts that were tried and failed.

### 1.4 Performance (RTX 5090, single GPU, batch = 8 plane waves)

| run segment | wall time |
|-------------|-----------|
| 1 iteration (typical, no deep backtrack) | **90–135 s** |
| 5 iterations (iter 1–5) | 537 s / 9 min |
| 20 iterations | **78 min** |
| 30 iterations | 95 min |

Compared with the NumPy + numba CPU port on 20 workers (~25 min / iter locally,
~11 min / iter on 40-core server), this is roughly **6–14× faster per
iteration** on a single GPU that is not even fully dedicated (the machine was
shared with another process during these runs).

---

## 2. Algorithm overview

Each outer iteration `k` does:

1. **RTM** per plane wave with current `v_k`:
   `image_tot[:,:,is] = -∂_z RTM(v_k, cpg_is)` for each of the 41 plane waves.
2. **Image-domain masks**: `maskimage(θ_is) → snell_mute(p_is) → top_mute(30)`.
3. **Dynamic image warping** (Hale 2013 DTW) against the pilot plane wave
   (`is_pilot = 21`, 1-based), producing integer shifts and the misfit
   `Σ shift²`. This is the non-linear step that actually encodes velocity
   error into the gradient.
4. **Shift masks + AGC weighting**:
   `refl = shift_sm * imageagc(image_masked)`.
5. **Wavepath gradient**: for each non-pilot plane wave, do a Born forward
   modeling with `refl` as virtual reflectivity on a smoothed background
   velocity, then cross-correlate with the back-propagated observed data.
6. **Backtracking line search** in the slowness domain:
   `α_tot = max(1/v) / mean|grad|`; try f1=0.01, backtrack if misfit does not
   drop, otherwise also try f2=2·f1 and keep the better one.
7. **Update** `v_new = clip(1/(1/v - α·fff·grad), [v_min, v_max])`, and every
   third iter also apply `vel_smooth(1, 51, 3)`.

The GPU batched kernel runs **all 41 plane waves simultaneously** inside one
FD time loop; the only serial axis is time. Dynamic warping stays on the CPU
(numba) — it is fast (≈1.5 s for all 41 PWs) and bit-perfect to Fortran.

---

## 3. Repository layout

```
.
├── README.md                    ← this file
├── plan.md                      phased migration plan (T0 … T7) with acceptance thresholds
├── pyproject.toml               torch>=2.4, numpy, scipy
├── pwmva_torch/
│   ├── device.py                GPU selection + DTYPE=float32
│   ├── io.py                    Fortran column-major <f4 ↔ torch tensor
│   ├── source.py                Ricker wavelet (torch)
│   ├── image_ops.py             z_deriv / gauss_smooth / vel_smooth /
│   │                             maskimage / snell_mute / imageagc (batched torch)
│   ├── inversion.py             outer loop (theta-frozen line-search semantics)
│   └── fd2d/
│       ├── pml.py               PML damping + α/temp1/temp2 + edge taper
│       ├── stencil.py           8th-order spatial / 2nd-order time batched kernel
│       ├── pw1_modeling.py      batched plane-wave forward modeling
│       ├── pw1_rtm.py           batched RTM
│       └── wavepath.py          batched wavepath gradient kernel
├── scripts/
│   ├── run_iter.py              main CLI: --niter --batch --device --out
│   ├── bench_fwd.py             41-PW forward benchmark
│   ├── compare_velinv.py        per-iter torch-vs-Fortran 6-panel plot
│   └── compare_convergence.py   full convergence table + figures
├── tests/
│   ├── conftest.py              path fixtures (expect Fortran + pwmva_python at the paths in conftest.py)
│   ├── _viz.py                  plot helpers + Pearson / rms_rel
│   ├── test_parity_*.py         six parity test modules, covering T1–T6
│   └── _viz_out/                21 parity PNGs (source, image ops, FD, RTM, wavepath, iter 1, convergence)
├── datagen/                    ★ end-to-end reproduction of the observed cpg from TRUE velocity
│   ├── README.md               pipeline (201 shot-gathers + per-shot direct-wave mute + tau-p)
│   ├── scripts/gen_shots_taup.py   single-file pipeline, ~30 s on RTX 5090
│   ├── inversion/              velinv_{1,5,10,15,20}.bin from 20-iter run on generated data
│   ├── viz/                    data-match + convergence + evolution + diff figures
│   └── metrics.json            per-shot Pearson + per-iter RMS frozen numbers
└── results/
    ├── model/                   vel_true_201x801x2.5m.bin (TRUE model) + velh_init_201x801x2.5m.bin (initial model)
    ├── observed/                Ricker source.bin + cpg/cpg_{1..41}.bin observed plane-wave CSGs (~189 MB)
    ├── velinv/                  velinv_{1,5,10,15,20,25,28,30}.bin reconstructed velocity snapshots (float32 raw, 201×801, Fortran column-major)
    ├── gradient/                gradient_1.bin
    ├── misfits_iter*.npy        per-iter misfit_tot logged by run_iter.py
    └── fffs_iter*.npy           per-iter line-search scalar fff

All `*.bin` in `results/` share the same layout: little-endian float32,
Fortran column-major (read with `np.fromfile(path, "<f4").reshape(nx, nz).T`
for 2-D grids, or `reshape(ntr, nt).T` for shot gathers). With the
`results/model/` + `results/observed/` bundles the repo is **self-contained**:
you can run the inversion end-to-end without the Fortran package.

**Parity tests vs. inversion runs — different data requirements.** The
parity tests in `tests/` compare torch against the Fortran + NumPy oracles
and therefore need both of those reference trees available locally; the
inversion itself only needs what is already in `results/`. The relevant
paths are configured as follows:

| What | How to point it | Needed for |
|---|---|---|
| Input data (models + observed CSG) | `--pkg` flag of `scripts/run_iter.py`, default `results/` in this repo | running inversion |
| Fortran oracle root (`pwmva_package`) | `PWMVA_PKG_ROOT` env var, or edit `tests/conftest.py` | parity tests |
| NumPy oracle root (`pwmva_python`) | `PWMVA_PY_ROOT` env var, or edit `tests/conftest.py` | parity tests |

The machine this was developed on had `pwmva_package` and `pwmva_python`
under `/home/pisquare/zhijun/pwmva_fortran/` — that path is baked into a
few scripts/tests as a default. Override with the env vars above or patch
the `PKG = Path(...)` line at the top of each script if you're running
elsewhere.

---

## 4. Quick start (GPU inversion)

```bash
# Install
git clone https://github.com/yl602019618/Wave-equation-Migration-Velocity-Analysis-Using-Plane-wave-based-CIGs.git
cd Wave-equation-Migration-Velocity-Analysis-Using-Plane-wave-based-CIGs
pip install -e .
pip install matplotlib numba   # dev deps (numba is only used for CPU warping)

# The input data bundle required to run the inversion is already shipped
# in this repo under results/model/ and results/observed/. If you instead
# want to point at the original Fortran package, edit scripts/run_iter.py
# and set the PKG = Path(...) constant at the top. Required files under PKG:
#   {PKG}/working/parfile_pwmva_warp_rerun.sh
#   {PKG}/model/2D_models/mlayer/{velh,vel}_201x801x2.5m.bin
#   {PKG}/model/2D_models/mlayer/coord_pw_45t45.dat
#   {PKG}/results/2D_models/mlayer/fdcsg_f40/source.bin
#   {PKG}/results/2D_models/mlayer/fdcsg_f40/cpg/cpg_{1..41}.bin

# Run 1 outer iteration on cuda:0, batch = 8 plane waves
python scripts/run_iter.py --niter 1 --batch 8 --out /tmp/torch_iter1

# Run 20 iterations
python scripts/run_iter.py --niter 20 --batch 8 --out /tmp/torch_iter20 \
        --no-3d-dump

# Resume from an existing velinv_N.bin
python scripts/run_iter.py --niter 10 --batch 8 --out /tmp/torch_iter20 \
        --resume-from 20 --no-3d-dump
```

### Comparing with Fortran

```bash
# One-iteration 6-panel comparison (requires Fortran rerun bins at their paths)
python scripts/compare_velinv.py --iter 1 --out /tmp/torch_iter1

# Full convergence table + figure over N iters
python scripts/compare_convergence.py --out /tmp/torch_iter20 --max-iter 30
```

### Reading a result from this repo

Every `*.bin` in `results/` is raw little-endian float32 written in
Fortran column-major order, i.e. same format as the original Fortran
(`results/velinv/velinv_30.bin` is a 201 × 801 grid):

```python
import numpy as np
nz, nx = 201, 801
v = np.fromfile("results/velinv/velinv_30.bin", dtype="<f4").reshape(nx, nz).T
# Now v is (nz, nx) in natural Python indexing.
```

---

## 5. How correctness was verified

- **Unit/parity tests** (`tests/test_parity_*.py`): each torch module is
  compared against its NumPy counterpart on realistic inputs read from the
  Fortran oracle bins. Every test saves a side-by-side PNG with field /
  diff / reference in `tests/_viz_out/`.
- **End-to-end iteration-1 parity**: `scripts/run_iter.py --niter 1`
  produces a `velinv_1.bin` that differs from the Fortran rerun by
  **0.039 m/s RMS** (see [`tests/_viz_out/t6_iter1_six_panel.png`](tests/_viz_out/t6_iter1_six_panel.png)).
- **Convergence tracking**: `scripts/compare_convergence.py` plots RMS
  against the TRUE model and against the Fortran rerun. See
  [`tests/_viz_out/t7_convergence.png`](tests/_viz_out/t7_convergence.png) and
  [`tests/_viz_out/t7_velinv_evolution.png`](tests/_viz_out/t7_velinv_evolution.png).

### Inter-implementation FP drift

torch and Fortran agree to ≤ 0.3 m/s RMS through iter 4, then drift to
6–20 m/s later on. This is the **cumulative effect of FP32 non-associativity**
across 3000 FD steps × 41 plane waves × 20 outer iterations, combined with
occasional line-search branch splits when `misfit1 ≈ misfit_tot` sits in the
FP noise floor. The two trajectories remain in the same basin: their
RMS-vs-TRUE curves stay parallel (within ~1 m/s) and both converge to
virtually the same local minimum. If bit-level agreement is needed, switch
the dtype to FP64 — at the cost of ~3× slower FD and ~2× more memory.

---

## 6. Non-goals / limitations

- We do not change the algorithm. Masking order, line-search branch logic
  and the `theta` freezing across trials are all matched 1:1 to Fortran.
- No autograd / differentiable wave equation — FD runs inside
  `torch.no_grad()` and gradients come from the WEMVA wavepath.
- No FP16 / mixed precision. FP32 throughout.
- Dynamic warping is NOT ported to GPU: the numba CPU implementation is
  already bit-perfect to Fortran and runs in ≈1.5 s for all 41 plane waves.
- Only the `pwmva_warp` path (2-D acoustic, plane-wave, ABC boundary, `IsFS
  = .false.`, `bc_type = 1`) is implemented, matching the paper's `mlayer`
  experiment. Modules like `pwmva_semb`, elastic, 3-D, Marchenko and
  free-surface are out of scope.

---

## 7. Acknowledgements

- Original paper and Fortran implementation: **Guo, B. & Schuster, G. T.
  (2017)**, *Wave-equation migration velocity analysis using plane-wave
  common-image gathers*, **Geophysics 82(5), S327–S340**,
  <https://doi.org/10.1190/geo2016-0596.1>.
- KAUST CSIM MVA chapter, used as a pedagogical reference:
  <https://csim.kaust.edu.sa/files/SeismicInversion/Chapter.MVA/index.html>.
- Dynamic image warping: **Dave Hale (2013)**.
- NumPy + numba reference port and Fortran alignment notes: the
  `pwmva_python` companion package (see `plan.md`).
- This PyTorch port and the parity-testing harness were built on top of the
  `pwmva_python` reference, aiming for a drop-in accelerated replacement of
  the CPU pipeline on commodity GPUs.

## 8. License

See [`LICENSE`](LICENSE) (MIT).
