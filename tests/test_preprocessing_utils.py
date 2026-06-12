import numpy as np
import pytest

from nicgiprep.utils.preprocessing_utils import (
    one_hot_encoding_with_gaussian,
    get_dct_basis_functions,
    projectKroneckerProductBasisFunctions,
    backprojectKroneckerProductBasisFunctions,
    computePrecisionOfKroneckerProductBasisFunctions,
    getGaussianLikelihoods,
    getGaussianPosteriors,
    undoLogTransformAndBiasField,
    fitBiasFieldParameters,
    bias_field_corr,
)


# ----------------------------------------
# tests for one_hot_encoding_with_gaussian
# ----------------------------------------
def test_one_hot_encoding_basic():
    target = np.array([[0, 1], [1, 0]])

    out = one_hot_encoding_with_gaussian(target, num_classes=2)

    assert out.shape == (2, 2, 2)
    assert out.dtype == int

    # class 0 positions
    assert np.all(out[..., 0][target == 0] == 1)
    assert np.all(out[..., 1][target == 1] == 1)


def test_one_hot_with_gaussian_smoothing():
    target = np.array([[0, 1], [1, 0]])

    out = one_hot_encoding_with_gaussian(target, num_classes=2, sigma=1.0)

    assert out.shape == (2, 2, 2)
    assert out.dtype == float
    assert np.isfinite(out).all()


# ----------------------------------------
# tests for get_dct_basis_functions
# ----------------------------------------
def test_dct_basis_shape():
    shape = (4, 5)
    basis = get_dct_basis_functions(shape, [2, 2])

    assert len(basis) == 2
    assert basis[0].shape[0] == 4
    assert basis[1].shape[0] == 5
    assert basis[0].shape[1] > 0
    assert basis[1].shape[1] > 0


# ----------------------------------------
# tests for projectKroneckerProductBasisFunctions + backprojectKroneckerProductBasisFunctions
# ----------------------------------------
def test_kron_project_backproject_roundtrip():
    # 2D tiny separable basis
    W1 = np.array([[1, 0], [0, 1]], dtype=float)

    W2 = np.array([[1, 0], [0, 1]], dtype=float)

    basis = [W1, W2]

    T = np.arange(4).reshape(2, 2)

    coeffs = projectKroneckerProductBasisFunctions(basis, T)
    recon = backprojectKroneckerProductBasisFunctions(basis, coeffs)

    assert recon.shape == T.shape
    np.testing.assert_allclose(recon, T, atol=1e-6)


# ----------------------------------------
# tests for computePrecisionOfKroneckerProductBasisFunctions
# ----------------------------------------
def test_precision_matrix_shape_and_symmetry():
    W = np.array([[1, 0], [0, 1]], dtype=float)

    basis = [W, W]
    B = np.ones((2, 2))

    H = computePrecisionOfKroneckerProductBasisFunctions(basis, B)

    assert H.shape[0] == H.shape[1]
    assert np.allclose(H, H.T, atol=1e-6)


# ----------------------------------------
# tests for getGaussianLikelihoods
# ----------------------------------------
# def test_gaussian_likelihood_shape():
#     data = np.array([[0.0, 1.0], [1.0, 2.0]]).T  # (2, 2)

#     mean = np.array([[0.0], [0.0]])

#     var = np.array([1.0, 1.0])

#     out = getGaussianLikelihoods(data, mean, var)

#     assert out.shape == (2, 2)
#     assert np.all(out > 0)


# ----------------------------------------
# tests for getGaussianPosteriors
# # ----------------------------------------
# def test_gaussian_posteriors_normalization():
#     data = np.array([[[0.0], [1.0]], [[1.0], [2.0]]])

#     priors = np.ones((2, 2)) * 0.5

#     means = np.zeros((2, 2, 1))
#     variances = np.ones((2, 2))

#     post, llh = getGaussianPosteriors(data, priors, means, variances)

#     assert post.shape == (2, 2)
#     assert np.allclose(post.sum(axis=1), 1.0, atol=1e-6)
#     assert np.isfinite(llh)


# ----------------------------------------
# tests for undoLogTransformAndBiasField
# ----------------------------------------
# def test_undo_log_transform():
#     img = np.log(np.ones((2, 2, 2, 1)) * 2)
#     bias = np.zeros_like(img)
#     mask = np.ones((2, 2, 2), dtype=bool)

#     out_img, out_bias = undoLogTransformAndBiasField(img, bias, mask)

#     assert out_img.shape == img.shape
#     assert np.allclose(out_img, np.ones_like(img), atol=1e-6)
#     assert np.all(out_bias > 0)


# ----------------------------------------
# tests for get_dct_basis_functions + fitBiasFieldParameters
# ----------------------------------------
def test_fit_bias_field_smoke():
    img = np.log(np.ones((2, 2, 2, 1)) * 2)

    seg = np.ones((2, 2, 2, 1))

    means = np.zeros((1, 1, 1))
    variances = np.ones((1, 1, 1))

    basis = get_dct_basis_functions((2, 2, 2), [2, 2, 2])

    coeff = fitBiasFieldParameters(
        img,
        seg.reshape(-1, 1),
        means,
        variances,
        basis,
        np.ones((2, 2, 2), dtype=bool),
        penalty=0.1,
    )

    assert coeff.shape[0] > 0
    assert np.isfinite(coeff).all()


# ----------------------------------------
# tests for bias_field_corr
# ----------------------------------------
# def test_bias_field_corr_smoke():
#     img = np.ones((3, 3, 3))
#     seg = np.ones((3, 3, 3, 1))

#     out_img, out_bias = bias_field_corr(
#         img, seg, penalty=0.0, patience=1, VERBOSE=False
#     )

#     assert out_img.shape == (3, 3, 3)
#     assert out_bias.shape == (3, 3, 3)
#     assert np.isfinite(out_img).all()
