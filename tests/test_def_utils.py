import os

import numpy as np
import nibabel as nib
import pytest
import torch

TEST_OUTPUT_DIR = "tests/fixtures"
IMG_SHAPE = (32, 32, 32)

from nicgiprep.utils.def_utils import (
    fast_3D_interp_torch,
    fast_3D_interp_field_torch,
    compute_gradient,
    compute_jacobian,
    lie_bracket,
    pole_ladder,
    svf_to_vox,
    svf_to_ras,
    SpatialTransformer,
    RescaleTransform,
)

"""
Tests for deepreg/download.py
pytest style
"""


# ----------------------------------------
# tests for fast_3D_interp_torch
# ----------------------------------------
def test_fast_interp_nearest():
    X = torch.arange(27).float().reshape(3, 3, 3)

    II = torch.tensor([[[0, 1], [1, 2]]]).float()
    JJ = torch.tensor([[[0, 1], [1, 2]]]).float()
    KK = torch.tensor([[[0, 1], [1, 2]]]).float()

    Y = fast_3D_interp_torch(X, II, JJ, KK, mode="nearest")

    assert Y.shape == II.shape
    assert torch.all(Y >= 0)


def test_fast_interp_linear_midpoint():
    X = torch.zeros((2, 2, 2))
    X[1, 1, 1] = 1.0

    II = torch.tensor([[[0.5]]])
    JJ = torch.tensor([[[0.5]]])
    KK = torch.tensor([[[0.5]]])

    Y = fast_3D_interp_torch(X, II, JJ, KK, mode="linear")

    assert Y.shape == (1, 1, 1)
    assert 0 < Y.item() < 1


def test_fast_interp_oob():
    X = torch.ones((3, 3, 3))

    II = torch.tensor([[[-1.0]]])
    JJ = torch.tensor([[[-1.0]]])
    KK = torch.tensor([[[-1.0]]])

    Y = fast_3D_interp_torch(X, II, JJ, KK, mode="linear")

    assert Y.item() == 0.0


# ----------------------------------------
# tests for fast_3D_interp_field_torch
# ----------------------------------------
def test_interp_field_channels():
    X = torch.zeros((3, 3, 3, 2))
    X[..., 0] = 1
    X[..., 1] = 2

    II = torch.tensor([[[1.0]]])
    JJ = torch.tensor([[[1.0]]])
    KK = torch.tensor([[[1.0]]])

    Y = fast_3D_interp_field_torch(X, II, JJ, KK)

    assert Y.shape[-1] == 2


def test_interp_field_nearest_channels():
    X = torch.rand((3, 3, 3, 2))

    II = torch.tensor([[[1.0]]])
    JJ = torch.tensor([[[1.0]]])
    KK = torch.tensor([[[1.0]]])

    Y = fast_3D_interp_field_torch(X, II, JJ, KK, mode="nearest")

    assert Y.shape[-1] == 2


# ----------------------------------------
# tests for compute_gradient
# ----------------------------------------
def test_gradient_zero_field():
    flow = np.zeros((5, 5, 5, 3))

    grad = compute_gradient(flow)

    assert grad.shape == (5, 5, 5, 3, 3)
    assert np.allclose(grad, 0)


def test_gradient_linear_field():
    x = np.arange(5)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    flow = np.stack([X, Y, Z], axis=-1)

    grad = compute_gradient(flow)

    # derivative of identity mapping ≈ identity
    I = np.eye(3)

    assert np.allclose(grad[2:-2, 2:-2, 2:-2], I, atol=1e-6)


# ----------------------------------------
# tests for compute_jacobian
# ----------------------------------------
def test_jacobian_identity():
    flow = np.zeros((5, 5, 5, 3))

    J = compute_jacobian(flow)

    assert J.shape == (5, 5, 5)
    assert np.allclose(J, 1.0, atol=1e-5)


# ----------------------------------------
# tests for lie_bracket
# ----------------------------------------
def test_lie_bracket_antisymmetry():
    v = np.random.randn(5, 5, 5, 3)
    w = np.random.randn(5, 5, 5, 3)

    vw = lie_bracket(v, w)
    wv = lie_bracket(w, v)

    assert np.allclose(vw, -wv, atol=1e-5)


# ----------------------------------------
# tests for pole_ladder
# ----------------------------------------
def test_pole_ladder_zero():
    v = np.zeros((5, 5, 5, 3))
    w = np.zeros((5, 5, 5, 3))

    out = pole_ladder(v, w, steps=5)

    assert np.allclose(out, 0)


# ----------------------------------------
# tests for svf_to_vox / svf_to_ras
# ----------------------------------------
def test_svf_roundtrip():
    svf = np.random.randn(5, 5, 5, 3)

    img = nib.Nifti1Image(svf, np.eye(4))

    vox = svf_to_vox(img)
    ras = svf_to_ras(vox)

    assert ras.shape == img.shape


# ----------------------------------------
# tests for SpatialTransformer
# ----------------------------------------
def test_spatial_transformer_identity():
    size = (5, 5, 5)
    transformer = SpatialTransformer(size)

    src = torch.rand((1, 1, *size))
    flow = torch.zeros((1, 3, *size))

    out = transformer(src, flow)

    assert torch.allclose(out, src, atol=1e-4)


# ----------------------------------------
# tests for RescaleTransform
# ----------------------------------------
def test_rescale_transform_downsample():
    x = torch.rand((1, 3, 10, 10, 10))

    layer = RescaleTransform((10, 10, 10), factor=0.5)

    y = layer(x)

    assert y.shape[2] < 10


def test_rescale_gaussian_kernel_exists():
    layer = RescaleTransform((10, 10), factor=0.5, gaussian_filter_flag=True)

    assert hasattr(layer, "kernel")
