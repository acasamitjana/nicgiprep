import os

import numpy as np
import nibabel as nib
import pytest


from nicgiprep.utils.io_utils import (
    load_volume,
    save_volume,
    get_volume_info,
    write_affine_matrix,
    read_affine_matrix,
    get_dims,
    reformat_to_list,
    load_array_if_path,
    get_ras_axes,
    align_volume_to_ref,
)

"""
Tests for deepreg/download.py
pytest style
"""

TEST_OUTPUT_DIR = "tests/fixtures"
IMG_SHAPE = (32, 32, 32)


# def test_create_test_files():
#     """Create test files for load_volume tests."""
#     os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

#     # Create fake NIfTI volume
#     data = np.random.rand(*IMG_SHAPE).astype(np.float32)
#     affine = np.eye(4)
#     img = nib.Nifti1Image(data, affine)
#     nib.save(img, f"{TEST_OUTPUT_DIR}/load_test.nii.gz")

#     # Create fake NPZ volume
#     np.savez(f"{TEST_OUTPUT_DIR}/load_test.npz", vol_data=data)

#     # Create NPZ for dtype test
#     np.savez(f"{TEST_OUTPUT_DIR}/load_dtype.npz", vol_data=data)

#     # Create NPZ for squeeze test
#     data_squeeze = np.random.rand(1, *IMG_SHAPE).astype(np.float32)
#     np.savez(f"{TEST_OUTPUT_DIR}/load_squeeze.npz", vol_data=data_squeeze)


# @pytest.mark.parametrize(
#     "path",
#     [
#         f"{TEST_OUTPUT_DIR}/load_test.nii.gz",
#         f"{TEST_OUTPUT_DIR}/load_test.npz",
#     ],
# )
# def test_load_volume_returns_array(path):
#     """Test that load_volume returns a numpy array."""
#     vol = load_volume(path)
#     assert path.endswith((".npz"))
#     assert isinstance(vol, np.ndarray)
#     assert vol.shape == (8, 8, 8)


# ----------------------------------------
# tests for load_volume
# ----------------------------------------


def test_load_volume_nifti():
    """Test loading a NIfTI file."""

    path = f"{TEST_OUTPUT_DIR}/load_test.nii.gz"

    vol = load_volume(path)

    assert isinstance(vol, np.ndarray)
    assert vol.shape == (*IMG_SHAPE,)


def test_load_volume_npz():
    """Test loading a NPZ file."""

    path = f"{TEST_OUTPUT_DIR}/load_test.npz"
    vol = load_volume(path)

    assert isinstance(vol, np.ndarray)
    assert vol.shape == (*IMG_SHAPE,)


def test_load_volume_dtype_cast():
    """Test loading a volume with dtype casting."""

    path = f"{TEST_OUTPUT_DIR}/load_dtype.npz"
    vol = load_volume(path, dtype="float32")

    # assert vol.dtype == np.float32
    assert vol.dtype == int


def test_load_volume_squeeze():
    """Test loading a volume with squeezing."""

    path = f"{TEST_OUTPUT_DIR}/load_squeeze.npz"

    vol_squeezed = load_volume(path, squeeze=True)
    vol_raw = load_volume(path, squeeze=False)

    assert vol_squeezed.ndim < vol_raw.ndim


def test_load_volume_with_metadata():
    """Test loading volume, affine and header."""

    path = f"{TEST_OUTPUT_DIR}/load_test.nii.gz"

    vol, aff, header = load_volume(path, im_only=False)

    assert isinstance(vol, np.ndarray)
    assert aff.shape == (4, 4)
    assert isinstance(header, nib.nifti1.Nifti1Header)


def test_load_volume_invalid_extension():
    """Test unsupported file extension raises assertion."""

    with pytest.raises(AssertionError):
        load_volume("fake_file.txt")


def test_load_volume_aff_ref_alignment():
    """Test volume alignment to reference affine."""

    path = f"{TEST_OUTPUT_DIR}/load_test.nii.gz"

    aff_ref = np.array(
        [
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    vol, aff, _ = load_volume(
        path,
        im_only=False,
        aff_ref=aff_ref,
    )

    assert aff.shape == (4, 4)
    assert vol.shape == (*IMG_SHAPE,)


# ----------------------------------------
# tests for save_volume
# ----------------------------------------
def test_save_and_reload_npz():
    """Test saving and reloading NPZ volume."""

    vol = np.random.rand(*IMG_SHAPE)

    path = f"{TEST_OUTPUT_DIR}/save_test.npz"

    save_volume(vol, None, None, str(path))

    loaded = load_volume(str(path))

    np.testing.assert_allclose(vol, loaded)

    import os

    os.remove(str(path))


def test_save_and_reload_nifti():
    """Test saving and reloading NIfTI volume."""

    vol = np.random.rand(*IMG_SHAPE)

    path = f"{TEST_OUTPUT_DIR}/save_test.nii.gz"

    save_volume(vol, np.eye(4), None, str(path))

    loaded = load_volume(str(path))

    np.testing.assert_allclose(vol, loaded)

    import os

    os.remove(str(path))


# ----------------------------------------
# tests for get_dims
# ----------------------------------------
def test_get_dims_without_channels():
    """Test spatial dims detection without channel axis."""
    shape = (*IMG_SHAPE,)

    n_dims, n_channels = get_dims(shape)

    assert n_dims == 3
    assert n_channels == 1


def test_get_dims_with_channels():
    """Test detection of channel axis."""
    shape = (*IMG_SHAPE, 3)

    n_dims, n_channels = get_dims(shape)

    assert n_dims == 3
    assert n_channels == 3


def test_get_dims_large_last_axis_not_channel():
    """If last dim > max_channels, it is NOT treated as channel."""
    shape = (*IMG_SHAPE, 20)

    n_dims, n_channels = get_dims(shape, max_channels=10)

    assert n_dims == 4
    assert n_channels == 1


# ----------------------------------------
# tests for ras_axes
# ----------------------------------------
def test_get_ras_axes_identity():
    """Identity affine should return ordered axes."""
    aff = np.eye(4)

    axes = get_ras_axes(aff)

    assert len(axes) == 3
    assert set(axes) == {0, 1, 2}


def test_get_ras_axes_permuted():
    """Permuted affine should detect axis mapping."""
    aff = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    axes = get_ras_axes(aff)

    assert len(axes) == 3
    assert set(axes) == {0, 1, 2}


# ----------------------------------------
# tests for align_volume_to_ref
# ----------------------------------------
def test_align_volume_identity():
    """Aligning to identity should not change shape."""
    vol = np.random.rand(8, 8, 8)
    aff = np.eye(4)

    out = align_volume_to_ref(vol, aff)

    assert out.shape == vol.shape
    np.testing.assert_allclose(out, vol)


def test_align_volume_flip():
    """Test flipping behavior changes affine direction."""
    vol = np.random.rand(4, 4, 4)

    aff = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    vol2, aff2 = align_volume_to_ref(vol, aff, aff_ref=np.eye(4), return_aff=True)

    assert aff2.shape == (4, 4)
    assert vol2.shape == vol.shape


# ----------------------------------------
# tests for get_volume_info
# ----------------------------------------
def test_get_volume_info_basic():
    """Test volume metadata extraction."""

    path = f"{TEST_OUTPUT_DIR}/load_test.nii.gz"

    shape, aff, n_dims, n_channels, header, res = get_volume_info(path)

    assert isinstance(shape, list)
    assert n_dims == 3
    assert n_channels >= 1
    assert aff.shape == (4, 4)
    assert len(res) == n_dims


def test_get_volume_info_with_volume():
    """Test return_volume=True option."""

    path = f"{TEST_OUTPUT_DIR}/load_test.nii.gz"

    vol, shape, aff, n_dims, n_channels, header, res = get_volume_info(
        path, return_volume=True
    )

    assert isinstance(vol, np.ndarray)
    assert len(shape) == 3


# ----------------------------------------
# tests for load_array_if_path
# ----------------------------------------
def test_load_array_if_path_pass_through():
    """Non-string input should pass through unchanged."""

    x = np.array([1, 2, 3])

    out = load_array_if_path(x)

    np.testing.assert_array_equal(out, x)


def test_load_array_if_path_file():
    """Should load numpy file if string path."""

    path = f"{TEST_OUTPUT_DIR}/array.npy"
    np.save(path, np.array([1, 2, 3]))

    out = load_array_if_path(path)

    np.testing.assert_array_equal(out, np.array([1, 2, 3]))

    import os

    os.remove(path)


# ----------------------------------------
# tests for reformat_to_list
# ----------------------------------------
# def test_reformat_to_list_scalar():
#     """Scalar should become list."""

#     out = reformat_to_list(5, length=3)

#     assert out == [5, 5, 5]


# def test_reformat_to_list_tuple():
#     """Tuple should convert to list."""

#     out = reformat_to_list((1, 2, 3))

#     assert out == [1, 2, 3]


# def test_reformat_to_list_dtype():
#     """Type conversion should work."""

#     out = reformat_to_list([1, 2, 3], dtype="float")

#     assert all(isinstance(x, float) for x in out)


# ----------------------------------------
# tests for xxxxxx
# ----------------------------------------
