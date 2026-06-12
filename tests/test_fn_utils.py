import os

import numpy as np
import nibabel as nib
import pytest

TEST_OUTPUT_DIR = "tests/fixtures"

from nicgiprep.utils.fn_utils import (
    align_with_identity_vox2ras0,
    rescale_volume,
    rescale_voxel_size,
    gaussian_antialiasing,
    one_hot_encoding,
    crop_label,
    compute_distance_map,
    compute_distance_map_nongrid,
    compute_centroids_ras,
    get_rigid_params,
)

"""
Tests for deepreg/download.py
pytest style
"""


# ----------------------------------------
# tests for align_with_identity_vox2ras0
# ----------------------------------------
def test_align_preserves_shape():
    V = np.random.rand(32, 40, 50)
    aff = np.eye(4)
    V2, aff2 = align_with_identity_vox2ras0(V, aff)
    assert V2.shape == V.shape
    assert aff2.shape == (4, 4)


def test_align_identity_affine_no_change():
    V = np.random.rand(10, 10, 10)
    aff = np.eye(4)
    V2, aff2 = align_with_identity_vox2ras0(V, aff)
    np.testing.assert_allclose(aff2, aff, atol=1e-6)


# ----------------------------------------
# tests for rescale_volume
# ----------------------------------------
def test_rescale_volume_range():
    vol = np.random.rand(20, 20, 20) * 100
    out = rescale_volume(vol, 0, 1)
    assert out.min() >= 0
    assert out.max() <= 1


def test_rescale_constant_volume():
    vol = np.ones((10, 10, 10))
    out = rescale_volume(vol)
    assert np.all(out == 0)


def test_rescale_percentile_clipping():
    vol = np.zeros((10, 10, 10))
    vol[0, 0, 0] = 1000
    out = rescale_volume(vol, min_percentile=0, max_percentile=100)
    assert np.isfinite(out).all()


# ----------------------------------------
# tests for rescale_voxel_size
# ----------------------------------------
def test_rescale_voxel_size_changes_shape():
    vol = np.random.rand(20, 20, 20)
    aff = np.eye(4)
    new_vol, new_aff = rescale_voxel_size(vol, aff, [2, 2, 2])

    assert new_vol.shape != vol.shape
    assert new_aff.shape == (4, 4)


def test_rescale_voxel_size_no_nans():
    vol = np.random.rand(10, 10, 10)
    aff = np.eye(4)
    new_vol, _ = rescale_voxel_size(vol, aff, [1.5, 1.5, 1.5])
    assert not np.isnan(new_vol).any()


# ----------------------------------------
# tests for gaussian_antialiasing
# ----------------------------------------
def test_gaussian_antialiasing_shape_preserved():
    vol = np.random.rand(10, 10, 10)
    aff = np.eye(4)
    out = gaussian_antialiasing(vol, aff, [2, 2, 2])
    assert out.shape == vol.shape


# ----------------------------------------
# tests for one_hot_encoding
# ----------------------------------------
def test_one_hot_basic():
    target = np.array([[[0, 1, 1, 0]]])  # .reshape(2, 2)
    out = one_hot_encoding(target)
    assert out.shape[-1] == 2
    assert np.all((out.sum(axis=-1) == 1))


def test_one_hot_preserves_labels():
    target = np.array([[0, 1], [1, 0]])
    out = one_hot_encoding(target)
    assert out[0, 0, 0] == 1


# ----------------------------------------
# tests for crop_label
# ----------------------------------------
def test_crop_label_contains_foreground():
    mask = np.zeros((20, 20, 20))
    mask[5:10, 5:10, 5:10] = 1

    cropped, coords = crop_label(mask, margin=2)

    assert cropped.sum() > 0
    assert len(coords) == 3


def test_crop_label_bounds_valid():
    mask = np.zeros((10, 10, 10))
    mask[2, 3, 4] = 1
    _, coords = crop_label(mask)

    for c in coords:
        assert c[0] < c[1]


# ----------------------------------------
# tests for apply_crop
# ----------------------------------------
def test_distance_map_shape():
    labelmap = np.zeros((10, 10, 10))
    labelmap[2:5, 2:5, 2:5] = 1

    out = compute_distance_map(labelmap)
    assert out.shape == (10, 10, 10, 2)


def test_distance_map_softmax():
    labelmap = np.random.randint(0, 2, (8, 8, 8))
    out = compute_distance_map(labelmap, soft_seg=True)
    assert np.allclose(out.sum(axis=-1), 1, atol=1e-4)


# ----------------------------------------
# tests for compute_distance_map_nongrid
# ----------------------------------------
def test_nongrid_distance_shape():
    labelmap = np.zeros((10, 10, 10))
    grid = np.stack(
        np.meshgrid(np.arange(10), np.arange(10), np.arange(10), indexing="ij")
    )

    out = compute_distance_map_nongrid(labelmap, grid)
    assert out.shape[:3] == (10, 10, 10)


# ----------------------------------------
# tests for compute_centroids_ras
# ----------------------------------------
def test_compute_centroids_shape():
    seg = np.zeros((10, 10, 10))
    seg[2:6, 2:6, 2:6] = 1

    nib.save(nib.Nifti1Image(seg, np.eye(4)), f"{TEST_OUTPUT_DIR}/seg.nii.gz")
    np.save(f"{TEST_OUTPUT_DIR}/labels.npy", np.array([1]))

    centroids, ok = compute_centroids_ras(
        f"{TEST_OUTPUT_DIR}/seg.nii.gz", f"{TEST_OUTPUT_DIR}/labels.npy"
    )

    assert centroids.shape[0] == 3
    assert len(ok) == 1

    # remove test files
    os.remove(f"{TEST_OUTPUT_DIR}/seg.nii.gz")
    os.remove(f"{TEST_OUTPUT_DIR}/labels.npy")


# ----------------------------------------
# tests for get_rigid_params
# ----------------------------------------
def test_rigid_params_output_shapes():
    matrix = np.eye(4)
    proxy = nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))

    angles, t = get_rigid_params(matrix, proxy)

    assert angles.shape == (3,)
    assert t.shape == (3,)


# ----------------------------------------
# tests for load_volume
# ----------------------------------------
def test_resample_does_not_break_volume_stats():
    vol = np.random.rand(20, 20, 20)
    aff = np.eye(4)

    v2, aff2 = rescale_voxel_size(vol, aff, [1.5, 1.5, 1.5])

    assert np.isfinite(v2).all()
    assert v2.size > 0
