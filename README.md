# Wave-equation Migration Velocity Analysis Using Plane-wave Common Image Gathers (PyTorch, GPU)

A PyTorch + CUDA reproduction of **Bowen Guo (2017)**, *"Wave-equation
Migration Velocity Analysis Using Plane-wave Common Image Gathers"*, KAUST.
Every FD / RTM / wavepath operation runs simultaneously over a batch of plane
waves on a single GPU. Correctness is validated step-by-step against both the
original Fortran implementation and a NumPy + numba CPU port.

- Original Fortran implementation: <https://github.com/> (Bowen Guo, KAUST,
  referenced as `pwmva_package` in the development notes)
- NumPy + numba reference implementation (CPU, bit-perfect to Fortran): the
  private `pwmva_python` package — see the paper notes in `plan.md`.
- Paper PDF (`Bowen.pdf`) available in the Fortran package `reference/`
  directory.

> If you are the author of the original Fortran code and would like this
> repository linked from yours, please open an issue.

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

### 1.3 Performance (RTX 5090, single GPU, batch = 8 plane waves)

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
└── results/
    ├── velinv/                  velinv_{1,5,10,15,20,25,28,30}.bin (int/float-32 raw, 201×801, Fortran column-major)
    ├── gradient/                gradient_1.bin
    ├── misfits_iter*.npy        per-iter misfit_tot logged by run_iter.py
    └── fffs_iter*.npy           per-iter line-search scalar fff
```

**Heads-up about the tests.** `tests/conftest.py` hard-codes two paths that
are local to the author's development machine:

- `pkg_root = /home/pisquare/zhijun/pwmva_fortran/pwmva_package` — the full
  Fortran code + data bundle (velocity model, observed CSG `cpg_*.bin`,
  reference outputs `velinv_<N>.bin`).
- `py_root = /home/pisquare/zhijun/pwmva_fortran/pwmva_python` — the NumPy
  + numba reference port whose `pwmva.*` modules are imported as oracles.

You will need both of those present to run the parity tests. If you only
want to run the **inversion itself**, `scripts/run_iter.py` does not depend
on any of that — it only needs the Fortran `pwmva_package` for the initial
velocity, observed data and parfile. Update the `PKG` constant at the top of
`scripts/run_iter.py` to point at your local copy.

---

## 4. Quick start (GPU inversion)

```bash
# Install
git clone https://github.com/yl602019618/Wave-equation-Migration-Velocity-Analysis-Using-Plane-wave-based-CIGs.git
cd Wave-equation-Migration-Velocity-Analysis-Using-Plane-wave-based-CIGs
pip install -e .
pip install matplotlib numba   # dev deps (numba is only used for CPU warping)

# Edit scripts/run_iter.py → set PKG to your pwmva_package path, or set up
# the same directory layout locally. Required files:
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

- Original Fortran implementation: **Bowen Guo** (KAUST, 2017).
- Dynamic image warping: **Dave Hale (2013)**.
- NumPy + numba reference port and Fortran alignment notes: the
  `pwmva_python` companion package (see `plan.md`).
- This PyTorch port and the parity-testing harness were built on top of the
  `pwmva_python` reference in collaboration with the author, aiming for a
  drop-in accelerated replacement of the CPU pipeline on commodity GPUs.

## 8. License

See [`LICENSE`](LICENSE) (MIT).
