# PWEMVA Torch GPU 加速复现计划

把 `pwmva_python`（已完成并与 Fortran bit-level 对齐）移植到 **PyTorch + CUDA**，在 **单/双 RTX 5090 (32 GB)** 上把一轮外迭代从 11–25 min 压到 **< 1 min**。

- 原 Fortran 仓库：`/home/pisquare/zhijun/pwmva_fortran/pwmva_package`
- NumPy+numba 参考实现（**gold oracle**）：`/home/pisquare/zhijun/pwmva_fortran/pwmva_python`
- 目标目录（本文件所在处）：`/home/pisquare/zhijun/pwmva_fortran/pwmva_torch`
- 论文：`/home/pisquare/zhijun/pwmva_fortran/Bowen.pdf`

---

## 0. 为什么用 Torch / 什么在 GPU 上

Python 版本耗时热点是两个 FD 主循环（41 个 plane wave × 3000 时间步 × 2 次传播 × 2 次 line-search trial）：

| 阶段 | 单 PW 耗时 (numpy+numba) | 占比 |
|------|--------------------------|------|
| RTM (`pw1_rtm`) | ~ 16 s | ~ 30% |
| Wavepath (`pw1_wavepath`) | ~ 28 s | ~ 55% |
| Warping (Hale DTW + numba) | 全部 41 PW **~1.5 s** | < 5% |
| 其他（image_ops, line search, misc） | - | < 10% |

结论：**FD 时间推进**是唯一值得上 GPU 的东西；warping 已经足够快，继续用 CPU+numba。

**批量化策略**：每个 plane wave 是独立的 FD 问题，只是 `p_ray`、`xref_1`、`seis`、`xg/zg` 不同——把它们合并成 **batch 维 B**，在 GPU 上一次推进 B 个独立波场。主循环中的 `for it in range(nt)` 仍然是串行的（时间不能向量化），但每一步 **同时推进 B 个 PW**，从而 B× 吞吐。

单 PW 波场 + 子采样存储的显存足迹（NZ=201, NX=801, NPML=100, NT=3000, dt_record=6, float32）：

| 张量 | 尺寸 | 字节 |
|------|------|------|
| FD 三缓冲 `p0/p1/p` (PML) | 3 × (401, 1001) × 4 | 4.8 MB |
| RTM 存 `wfs` / `wfr`（物理网格，nt_record ≈ 501） | 2 × (201, 801, 501) × 4 | 646 MB |
| Wavepath 存 `pp_store` / `ppb_store` | 2 × (201, 801, 501) × 4 | 646 MB |

→ **B = 10** 个 PW 并行：wavepath 需要 ~ 6.5 GB；加上 PML 常量张量和 seis，≈ 8 GB，**1 块 5090 可跑**。B = 20 则 ~ 13 GB，已可跑当前自由显存（~15 GB）。

两块 GPU 时把 ns=41 拆成 [21,20]，每张卡并行处理自己的一半，再 `all_reduce`(sum) 汇总梯度。**MVP 先单卡批处理跑通**，多卡放在 T8 后期优化。

---

## 1. 算法回顾（与论文、与 Python 一致，无任何改动）

每个 outer-iteration `iii`（目标 20 轮）：

1. **RTM**：`image_tot[:,:,is] = -∂_z RTM(v, cpg_is)` 共 41 个 plane wave。
2. **像域 mask**：`maskimage(θ_is) → snell_mute(p_is) → top_mute(30)`。
3. **Warping** (Hale 2013 DTW)：以 `is_pilot=20` (0-based) 的零角度像为 pilot，得到 `shift[:,:,is]` 和 `misfit_tot = Σ shift²`。
4. **Shift mask**：`top_mute → maskimage → snell_mute`；记录 `num[is] = nzmean2d(shift_masked)`。
5. **shift 平滑 + AGC 权重**：`shift_sm = gaussian_smooth(5, 20, shift)`；`refl[:,:,is] = shift_sm[:,:,is] * imageagc(image_masked[:,:,is])`。
6. **背景速度平滑**：`v_smooth = vel_smooth(v, 11, 21, 3)`。
7. **Wavepath 梯度核**：对每个非 pilot PW，以 `refl` 为虚拟反射率做 Born 正演，再与观测 `cpg` 做互相关成像 → `g[is]`；`gk1 += g[is]/max|g[is]| * num[is]`。
8. **梯度后处理**：`dk1[:20,:]=0`；`dk1 = gaussian_smooth(2, 20, dk1)`。
9. **Backtracking line search**（slowness 域）：`alpha_tot = max(s)/mean|grad|`，`s=1/v`；试 `f1=0.01`，若不降 misfit → 折半；否则试 `f2=2*f1` 取优。每次 trial 重跑 step 1–3。
10. **更新**：`v = clip(1/(s - alpha·fff·grad), [vmin,vmax])`；若 `iii % 3 == 1` 再做 `vel_smooth(1, 51, 3)`。

Parfile 参数（`parfile_pwmva_warp_rerun.sh`）：
`NX=801, NZ=201, DX=2.5, NT=3000, DT=0.0006, FREQ=40, NW=3000, NPML=100, VMIN=1500, VMAX=2000, NIT=20, INV_TYPE=SD`；41 plane waves (`coord_pw_45t45.dat`)，拾 `is=21`（1-based）为 pilot。

---

## 2. 目标目录结构

```
pwmva_torch/
├── plan.md                         ← 本文件
├── readme.md                       （后期写）
├── pyproject.toml                  依赖：torch>=2.4, numpy, scipy, numba (复用 warping)
├── pwmva_torch/                    库代码
│   ├── __init__.py
│   ├── device.py                   device 选择 / 多卡 helper / dtype (float32) / torch.set_num_threads
│   ├── io.py                       复用 pwmva_python 的 raw <f4 读写（张量版）
│   ├── config.py                   复用 pwmva_python.config 直接 import（parfile 解析）
│   ├── geom.py                     复用 pwmva_python.geom（纯 numpy 数据结构）
│   ├── source.py                   torch.Tensor Ricker
│   ├── image_ops.py                torch 版 z_derivative / gaussian_smooth (conv1d) /
│   │                               vel_smooth (cumsum prefix-sum) / maskimage /
│   │                               snell_mute / top_mute / imageagc / nzmean2d / mean_abs
│   ├── warping.py                  **复用 pwmva_python.warping**（numba CPU，已经 1.5s/轮）
│   ├── fd2d/
│   │   ├── __init__.py
│   │   ├── pml.py                  torch 版 abc_get_damp2d / build_alpha_temp / build_taper
│   │   ├── stencil.py              8 阶空间 Laplacian + 时间推进 kernel（支持 batch 维）
│   │   ├── pw1_modeling.py         批量平面波正演（vmap over B plane waves）
│   │   ├── pw1_rtm.py              批量 RTM（source forward + receiver back + 互相关）
│   │   └── wavepath.py             批量 wavepath 梯度核（background + Born + receiver back）
│   ├── inversion.py                外循环：去掉 ProcessPoolExecutor，改成 GPU 批量调用
│   └── check.py                    CLI：对任意 velinv_<N>.bin 比对 numpy 版本
├── scripts/
│   ├── run_iter.py                 CLI：--niter --device cuda:0 --batch 10 --out …
│   ├── bench.py                    单函数 benchmark（py vs torch CPU vs torch GPU）
│   └── compare.py                  结果对比脚本（RMS / Pearson / viz）
└── tests/
    ├── conftest.py                 固定 Fortran + Python oracle 路径
    ├── _viz.py
    ├── test_parity_io.py           T0：io / config / geom 解析无歧义
    ├── test_parity_source.py       T1：Ricker torch == numpy allclose
    ├── test_parity_image_ops.py    T2：每个 op vs numpy allclose(rtol=1e-5)
    ├── test_parity_fd_modeling.py  T3：单 PW → cpg_full / batch 4 PW 的 B 元素 == 单独算
    ├── test_parity_rtm.py          T4：image_tot_1.bin Pearson > 0.9999
    ├── test_parity_wavepath.py     T5：gradient_1.bin Pearson > 0.9999
    ├── test_parity_iter1.py        T6：velinv_1.bin RMS < 1 m/s
    └── test_parity_iter5.py        T7：5 轮 RMS < 5 m/s（slow test，跳过默认）
```

---

## 3. 迁移阶段（自底向上，每一阶段对齐 Python oracle 后方可前进）

每个阶段必须同时满足：(a) 正确性对齐阈值；(b) 性能达标（如有目标）。

| Phase | 模块 | 输入 oracle | 验收准则 (正确性) | 性能目标 |
|-------|------|------------|--------------------|----------|
| **T0** | `io.py` / `config.py` / `geom.py` / `device.py` | 原始 bin + parfile + coord_pw_45t45.dat | 读出 tensor 与 numpy 数组 `allclose` | — |
| **T1** | `source.py` (Ricker) | `source.bin` | max\|Δ\| < 1e-6 | — |
| **T2** | `image_ops.py` | 任取 `shift_1.bin` / `image_tot_1.bin`，运行 Python 版当输入 oracle | 每个 op `allclose(rtol=1e-4, atol=1e-5)` vs numpy 版本 | gaussian_smooth 单次 < 1 ms on GPU |
| **T3** | `fd2d/pml.py` + `fd2d/stencil.py` + `fd2d/pw1_modeling.py` | `cpg_full_<is>.bin` (Fortran oracle) | vs Fortran 峰值振幅一致 5 位；批量 B=10 结果 == 单独 loop 算 41 次 | 41 PW 合成数据 < 5 s on 1×5090（CPU 基线 ~10 min） |
| **T4** | `fd2d/pw1_rtm.py` | `image_tot_1.bin`（21 / 11 / 31 三个 PW 抽查） | Pearson ≥ 0.9999，RMS rel < 5%；`image_tot_1.bin` 整盒 Pearson ≥ 0.9999 | 41 PW RTM < 10 s on GPU |
| **T5** | `fd2d/wavepath.py` | `gradient_1.bin` | 端到端 `gradient_1.bin` Pearson ≥ 0.9999，rms_rel < 2% | 40 PW wavepath < 15 s on GPU |
| **T6** | `inversion.py` (单轮) | `velinv_1.bin` | RMS vs Fortran rerun < 1 m/s（与 Python 单轮 RMS 0.039 同量级±FP 漂移） | **单轮 < 60 s on 1×5090** |
| **T7** | 多轮外循环 | `velinv_5.bin` / `velinv_10.bin` | 与 Python 连续轮 RMS 漂移同量级 (iter 5: < 10 m/s)，`vs true` 同步下降 | 20 轮 < 30 min |
| **T8**（可选） | 双 GPU 分片 | 同上 | 结果与单卡一致（sum-reduce 顺序 tolerance） | 20 轮 < 15 min |

---

## 4. 对齐策略细节

### 4.1 数值等价性的阶梯

GPU FP32 求和顺序不同于 CPU numpy（OpenMP 并行归约、cuBLAS reduction），所以**不追求 bit-level 一致**；追求 **Pearson ≥ 0.9999，RMS 相对误差 < 1–2%**，与 Python 论文中提到的"跨机器跨 numpy 版本 FP 漂移"同性质。

- `atol/rtol=0`：I/O、Ricker、整数 mask 索引
- `rtol ≈ 1e-5`：单步 stencil、gaussian_smooth、vel_smooth prefix sum
- `rtol ≈ 1e-3`：单 PW RTM / wavepath
- `rms_rel < 1%`：iter 1 端到端梯度
- `RMS < 1 m/s`：iter 1 velinv

**双 oracle**：既比 Python numpy 输出，也比 Fortran 原始输出。新 FD 上 GPU 后，如果与 Python 对齐但与 Fortran 偏差大，属于已知跨实现 FP 漂移（见 readme P3 节），可接受；如果与 Python 都对不齐，说明移植出 bug。

### 4.2 Batched FD stencil 的两种候选实现

1. **`torch.roll` + slice（推荐起步）**
   - 直接把 numpy 的 `p1[iz0:iz1, ix0+1:ix1+1] + ...` 翻译成 torch slicing，多加一个 batch 维 `p[B, nz_pml, nx_pml]`。
   - 在 5090 上预期 stencil-bound，吞吐约 600–1000 GFLOP/s，够用。
   - **风险**：slicing 生成临时张量，显存抖动。用 `torch.compile` 或合并 `aten.add` 可缓解。

2. **`F.conv2d` with 9×9 kernel**
   - 用预置的 8 阶 Laplacian 9×9 kernel 作 conv2d；cuDNN 会把它映射到高效 implicit-GEMM。
   - 但 cuDNN 对非 3×3 kernel 在小 batch 时反而慢；先不走这条路。

3. **Triton kernel**（只在前两者达不到目标时才写）

### 4.3 源/接收子的高效注入

- **Plane-wave 源**：`src_line[B, nx]` 预先构造好（nt × B × nx 的 shift table 太大），**每步**按 `itt = it - delays` 索引从 `src[nw]` 取值；delays 是 `(B, nx)` int64，vectorized `gather` 一步完成。
- **接收子注入**：seis 是 `(B, nt, ng)`，每步 `p[B, igz, igx] += beta * seis[:, it-1, :]`；igz/igx 每个 PW 不同，用 `torch.scatter_add_` 扁平化后加。

### 4.4 子采样存储

Fortran/Python 行为：`it_record = round(it / 6)`；每一步都覆盖式写（最后写入获胜，Python 已复刻此行为）。
Torch 版直接照抄：对存储张量 `wfs[B, nz, nx, nt_record]` 按 `it_record` 索引切片赋值。

**显存优化**：如果 B × nt_record × nz × nx 显存吃紧，可以：
- 降 B 到 10 或 8，分 5 次跑 41 PW
- 只保留物理网格（已经在做）
- `dt_record=6` 已经是 Fortran 硬编码，不能改

### 4.5 Warping 继续跑 CPU

`pwmva_python/warping.py` 的 numba `@njit(parallel=True)` 每轮只要 1.5 s，没必要上 GPU。**直接 import 使用**，把 image_tot 从 GPU `.to("cpu").numpy()` 喂进去，拿回 shift 后 `.to(device)` 即可（Only 200×801×41×4 = 26 MB 往返）。

### 4.6 Line search 与 trial RTM

Line search 的 `trial_misfit` 需要**重做 RTM + warping**。Python 版串行 trial 1/2/3 合计 ~ 7 分钟。GPU 版预计每 trial ~ 10 s，整个 line search < 1 min。

### 4.7 与 Python 版共享的基础设施

能直接 import 不动的：
- `pwmva_python.config.read_parfile`
- `pwmva_python.geom.read_pw1_coord`
- `pwmva_python.warping.warping_misfit`

需要重写（torch 版本）：`io`（tensor 直读）、`source`、`image_ops`、`fd2d/*`、`inversion`。

---

## 5. 风险点与对策

1. **显存爆**：B×nt_record×nz×nx 超 32 GB。对策：B=10 起步，用 `torch.cuda.memory._record_memory_history` 查峰值；必要时把 `wfr` 改成只存部分时段。
2. **PML FP 漂移**：`build_alpha_temp` 在 torch FP32 下与 numpy 有 ULP 差异（`**2` 顺序）。对策：按 Python 的顺序逐项翻译，必要时中间步转 FP64 再回 FP32。
3. **Snell mute / argmax 差异**：torch `argmax` 在 ties 时行为与 numpy 可能不同。对策：写成 `torch.where(cond.any(0), cond.int().argmax(0), nz)` 并对照 numpy cases 单测。
4. **`imageagc` FFT**：torch.fft 与 scipy.fftconvolve 在 edge padding 上等价（linear conv，长度 `n11+np_pad-1`）。对策：用 `torch.nn.functional.conv1d` with zero-pad + slicing 而不是 FFT，对几十的 kernel 更快且更可控。
5. **多卡 reduce 顺序**：双 GPU 对 `gk1_tot += Σ g[is]/max|g[is]|*num[is]` 做 reduce 时顺序不固定，可能造成 FP 漂移。对策：先不上多卡；上多卡后对每 PW 结果按 ipro 排序后再求和。
6. **Line-search 决策分叉**：当 misfit1 与 misfit_tot 差距在 FP 噪声水平，可能走不同分支（Python 已经看到过，iter 7 fff=4.88e-6）。对策：接受；论文目标是 `vs true RMS` 同步下降，不是与 Fortran 每轮 bit 等价。
7. **torch.compile**：2.11 上 torch.compile + CUDA graph 对小 stencil 可能有 10–30% 收益，但编译 warmup 5–10 s。先裸跑，后期加上。
8. **非确定性 kernel**：`torch.backends.cudnn.deterministic = True` 只影响 conv；scatter_add 自身非确定。对策：若需完全可复现，用 `torch.use_deterministic_algorithms(True)` 但会变慢；正常开发不强求。

---

## 6. 里程碑

- **M1**：T0–T2 通过 → pyproject 可 pip install -e，基础算子 parity
- **M2**：T3 通过 → 单卡 41 PW modeling < 5 s，与 Fortran/Python 对齐
- **M3**：T4+T5 通过 → `gradient_1.bin` Pearson > 0.9999，单轮 FD 部分 < 30 s
- **M4**：T6 通过 → 单轮外迭代 `velinv_1.bin` RMS < 1 m/s，**单轮 < 60 s**（**~10× 加速**）
- **M5**：T7 通过 → 20 轮 < 30 min，与 Python 收敛曲线一致
- **M6**（可选）：T8 双卡 → 20 轮 < 15 min

---

## 7. 不做的事

- **不改变算法**：保留论文 + Python 版完全一致的外循环、line search 策略、mask 顺序。
- **不做 autograd**：全程 `torch.no_grad()`（我们是手写物理梯度，不用反向传播）。
- **不做半精度 / 混合精度**：FP32 全程，避免把 FD 弄散。
- **不替换 warping 算法**：numba CPU 版本已经 bit-perfect 对齐 Fortran，换 GPU 版本只会引入 parity 风险却几乎没有收益。
- **不复刻 `pwmva_semb` / 3D / 弹性 / `isFS=true` / `bc_type=2`**（与 Python 版保持一致）。
- **不构建独立 I/O 层**：二进制格式与 Fortran 完全相同（raw float32，Fortran column-major），直接复用 Python io 接口即可。

---

## 8. 参考对照表（关键数据和路径）

| 文件 | 路径 |
|------|------|
| Fortran package 根 | `/home/pisquare/zhijun/pwmva_fortran/pwmva_package` |
| parfile | `…/working/parfile_pwmva_warp_rerun.sh` |
| 初始速度 `velh` | `…/model/2D_models/mlayer/velh_201x801x2.5m.bin` |
| 真实速度 `vel` | `…/model/2D_models/mlayer/vel_201x801x2.5m.bin` |
| Plane-wave 几何 | `…/model/2D_models/mlayer/coord_pw_45t45.dat`（41 PW × 401 recv） |
| Source (Ricker, 40 Hz) | `…/results/2D_models/mlayer/fdcsg_f40/source.bin` |
| 观测 CSG `cpg_<is>.bin` | `…/results/2D_models/mlayer/fdcsg_f40/cpg/` |
| Oracle 中间产物 | `…/results/2D_models/mlayer/pwmva_warp_rerun/` |
| Full-FD oracle | `…/results/2D_models/mlayer/cpg_oracle/cpg_full_<is>.bin` |
| Python 参考实现 | `/home/pisquare/zhijun/pwmva_fortran/pwmva_python/pwmva/` |

---

## 9. 性能预算汇总（单卡 RTX 5090，初步估计）

| 阶段 | 单轮时间估计 |
|------|------------|
| RTM 41 PW，B=10 | ~ 8 s |
| 2 × line-search trial RTM (非 pilot 40 PW × 2) | ~ 15 s |
| Wavepath 40 PW，B=10 | ~ 15 s |
| image_ops + warping (CPU, 不变) | ~ 3 s |
| I/O + 杂项 | ~ 2 s |
| **单轮总计** | **~ 45 s** |
| 20 轮 | **~ 15 min** |

相对 Python 远程 40-worker（~ 11 min/轮）的 **~15×** 加速，相对本地 20-worker（~ 25 min/轮）的 **~33×** 加速。

如实际达成 < 30 s/轮（通过 torch.compile + 双卡），20 轮可压到 **< 10 min**。
