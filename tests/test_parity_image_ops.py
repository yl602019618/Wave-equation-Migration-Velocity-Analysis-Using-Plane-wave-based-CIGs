"""T2 parity: image-domain operators torch vs numpy (pwmva_python)."""
from pathlib import Path
import numpy as np
import torch
import pytest

import pwmva_torch.image_ops as ops_t
from pwmva_torch.device import pick_device
from pwmva_torch import io as io_t

# numpy reference
import pwmva.image_ops as ops_np
from pwmva.io import read2d as read2d_np, read3d as read3d_np

from _viz import plot_2d_compare, pearson, rms_rel

DEVICE = pick_device()


def _to_np(t):
    return t.detach().to("cpu").numpy()


def test_z_derivative_2d(rerun_dir, viz_dir):
    img = read2d_np(rerun_dir / "gradient_1.bin", 201, 801).astype(np.float32)
    out_np = ops_np.z_derivative_2d(img, fd_order=4)
    out_t  = ops_t.z_derivative_2d(torch.from_numpy(img).to(DEVICE), fd_order=4)
    out_t  = _to_np(out_t)

    rms = rms_rel(out_np, out_t); p = pearson(out_np, out_t)
    print(f"\n[T2/z_deriv] pearson={p:.6f}  rms_rel={rms:.3e}")
    assert p > 0.99999
    assert rms < 1e-5
    plot_2d_compare(out_np, out_t, ["np z_deriv", "torch z_deriv", "diff"],
                    viz_dir / "t2_z_deriv.png")


def test_gaussian_smooth_2d(rerun_dir, viz_dir):
    """Compare gaussian_smooth(5, 20) on shift_<iii>.bin — the step Fortran runs in the
    main loop. We pick iter 1 to keep it cheap."""
    shift = read3d_np(rerun_dir / "shift_1.bin", 201, 801, 41)
    h = shift[:, :, 21].astype(np.float32)    # some non-pilot plane wave

    out_np = ops_np.gaussian_smooth_2d(5.0, 20.0, h)
    out_t  = _to_np(ops_t.gaussian_smooth_2d(5.0, 20.0,
                                             torch.from_numpy(h).to(DEVICE)))
    rms = rms_rel(out_np, out_t); p = pearson(out_np, out_t)
    print(f"\n[T2/gauss2d] pearson={p:.6f}  rms_rel={rms:.3e}  "
          f"max|np|={np.abs(out_np).max():.3e}")
    assert p > 0.9999
    assert rms < 1e-4
    plot_2d_compare(out_np, out_t, ["np gauss(5,20)", "torch gauss(5,20)", "diff"],
                    viz_dir / "t2_gauss_smooth.png")


def test_vel_smooth(model_dir, viz_dir):
    velh = read2d_np(model_dir / "velh_201x801x2.5m.bin", 201, 801).astype(np.float32)
    # velh is constant 1500 — add a synthetic layer so smoothing does something
    v = velh.copy()
    v[100:, :] = 2000.0
    out_np = ops_np.vel_smooth(v, nzs=11, nxs=21, niter=3)
    out_t  = _to_np(ops_t.vel_smooth(torch.from_numpy(v).to(DEVICE),
                                     nzs=11, nxs=21, niter=3))
    rms = rms_rel(out_np, out_t); p = pearson(out_np, out_t)
    print(f"\n[T2/vel_smooth] pearson={p:.7f}  rms_rel={rms:.3e}")
    assert p > 0.999999
    assert rms < 1e-5
    from _viz import plot_2d_jet_compare
    plot_2d_jet_compare(out_np, out_t, ["np vel_smooth", "torch", "diff"],
                        viz_dir / "t2_vel_smooth.png")


def test_maskimage_snell_mute_top_mute(rerun_dir, viz_dir):
    img_all = read3d_np(rerun_dir / "image_tot_1.bin", 201, 801, 41).astype(np.float32)
    img = img_all[:, :, 10].copy()

    # maskimage
    theta = 0.3
    m_np = ops_np.maskimage(img, theta, 30)
    m_t  = _to_np(ops_t.maskimage(torch.from_numpy(img).to(DEVICE), theta, 30))
    assert np.allclose(m_np, m_t, atol=0, rtol=0), "maskimage must be bit-identical"

    # top_mute
    m_np = ops_np.top_mute(img, 30)
    m_t  = _to_np(ops_t.top_mute(torch.from_numpy(img).to(DEVICE), 30))
    assert np.allclose(m_np, m_t, atol=0, rtol=0)

    # snell_mute — needs a slowness field (1/v)
    v = 1500.0 + 300.0 * np.random.default_rng(0).random((201, 801)).astype(np.float32)
    slow = 1.0 / v
    m_np = ops_np.snell_mute(img, 1.5e-4, slow)
    m_t  = _to_np(ops_t.snell_mute(torch.from_numpy(img).to(DEVICE), 1.5e-4,
                                   torch.from_numpy(slow).to(DEVICE)))
    # snell_mute's argmax/where ties should align exactly in integer indexing
    assert np.allclose(m_np, m_t, atol=0, rtol=0), "snell_mute should be bit-identical"
    print("\n[T2/masks] maskimage / top_mute / snell_mute all bit-identical.")


def test_imageagc(rerun_dir, viz_dir):
    img_all = read3d_np(rerun_dir / "image_tot_1.bin", 201, 801, 41).astype(np.float32)
    img = img_all[:, :, 10]   # some non-pilot plane wave
    out_np = ops_np.imageagc(img, np_pad=30)
    out_t  = _to_np(ops_t.imageagc(torch.from_numpy(img).to(DEVICE), np_pad=30))

    rms = rms_rel(out_np, out_t); p = pearson(out_np, out_t)
    print(f"\n[T2/imageagc] pearson={p:.6f}  rms_rel={rms:.3e}")
    # imageagc amplifies small values via sqrt(|·|)+1e-4, so FP32 FFT ordering
    # differences between scipy and torch.fft inflate rms_rel even though the
    # output is nearly identical in Pearson / structure.
    assert p > 0.9999
    assert rms < 5e-3
    plot_2d_compare(out_np, out_t, ["np imageagc", "torch imageagc", "diff"],
                    viz_dir / "t2_imageagc.png")


def test_scalars(rerun_dir):
    img_all = read3d_np(rerun_dir / "image_tot_1.bin", 201, 801, 41)
    img = img_all[:, :, 10].astype(np.float32)
    ma_np  = ops_np.mean_abs(img); ma_t = ops_t.mean_abs(torch.from_numpy(img).to(DEVICE))
    nz_np  = ops_np.nzmean2d(img); nz_t = ops_t.nzmean2d(torch.from_numpy(img).to(DEVICE))
    assert abs(ma_np - ma_t) / (abs(ma_np) + 1e-30) < 1e-5
    assert abs(nz_np - nz_t) / (abs(nz_np) + 1e-30) < 1e-5
    print(f"\n[T2/scalars] mean_abs np={ma_np:.6e} t={ma_t:.6e}  "
          f"nzmean np={nz_np:.6e} t={nz_t:.6e}")
