from typing import Optional, Union, Sequence, Tuple
from pathlib import Path
import nibabel as nib

import numpy as np
import surfa as sf
from nibabel import Nifti1Image
from scipy.optimize import linprog
import torch
import torch.nn as nn
import torch.nn.functional as F

from nicgiprep.utils.fn_utils import compute_centroids_ras
from scipy.ndimage import gaussian_filter

def fast_3D_interp_torch(
    X: torch.Tensor, II: torch.Tensor, JJ: torch.Tensor, KK: torch.Tensor, mode: str
) -> torch.Tensor:
    """Interpolate a 3D scalar volume at arbitrary coordinates using PyTorch.

    Parameters
    ----------
    X : torch.Tensor
        Source volume, shape ``(Nx, Ny, Nz)``.
    II : torch.Tensor
        x-coordinates of the query points (first dimension), shape (Rx, Ry, Rz).
    JJ : torch.Tensor
        y-coordinates of the query points (second dimension), shape (Rx, Ry, Rz).
    KK : torch.Tensor
        z-coordinates of the query points (third dimension), shape (Rx, Ry, Rz).
    mode : {'nearest', 'linear'}
        Interpolation mode. ``'linear'`` uses trilinear interpolation;
        out-of-bounds coordinates are set to zero.

    Returns
    -------
    torch.Tensor
        Interpolated values, shape ``(*II.shape, C)`` == ``(Rx, Ry, Rz, C)``.

    Raises
    ------
    Exception
        If ``mode`` is not ``'nearest'`` or ``'linear'``.
    """
    if mode == "nearest":
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = X.shape[0] - 1
        JJr[JJr > (X.shape[1] - 1)] = X.shape[1] - 1
        KKr[KKr > (X.shape[2] - 1)] = X.shape[2] - 1
        Y = X[IIr, JJr, KKr]

    elif mode == "linear":
        ok = (
            (II >= 0)
            & (JJ >= 0)
            & (KK >= 0)
            & (II <= X.shape[0] - 1)
            & (JJ <= X.shape[1] - 1)
            & (KK <= X.shape[2] - 1)
        )
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]
        #
        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = X.shape[0] - 1
        wcx = IIv - fx
        wfx = 1 - wcx
        #
        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = X.shape[1] - 1
        wcy = JJv - fy
        wfy = 1 - wcy
        #
        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = X.shape[2] - 1
        wcz = KKv - fz
        wfz = 1 - wcz
        #
        c000 = X[fx, fy, fz]
        c100 = X[cx, fy, fz]
        c010 = X[fx, cy, fz]
        c110 = X[cx, cy, fz]
        c001 = X[fx, fy, cz]
        c101 = X[cx, fy, cz]
        c011 = X[fx, cy, cz]
        c111 = X[cx, cy, cz]
        #
        c00 = c000 * wfx + c100 * wcx
        c01 = c001 * wfx + c101 * wcx
        c10 = c010 * wfx + c110 * wcx
        c11 = c011 * wfx + c111 * wcx
        #
        c0 = c00 * wfy + c10 * wcy
        c1 = c01 * wfy + c11 * wcy
        #
        c = c0 * wfz + c1 * wcz
        #
        Y = torch.zeros(II.shape, device=X.device)
        Y[ok] = c.float()

    else:
        raise Exception("mode must be linear or nearest")

    return Y


def fast_3D_interp_field_torch(
    X: torch.Tensor,
    II: torch.Tensor,
    JJ: torch.Tensor,
    KK: torch.Tensor,
    mode: str = "linear",
) -> torch.Tensor:
    """Interpolate a 3D multi-channel field at arbitrary coordinates using PyTorch.

    Parameters
    ----------
    X : torch.Tensor
        Source field, shape ``(Nx, Ny, Nz, C)``.
    II : torch.Tensor
        x-coordinates of the query points (first dimension), shape (Rx, Ry, Rz).
    JJ : torch.Tensor
        y-coordinates of the query points (second dimension), shape (Rx, Ry, Rz).
    KK : torch.Tensor
        z-coordinates of the query points (third dimension), shape (Rx, Ry, Rz).
    mode : {'nearest', 'linear'}, optional
        Interpolation mode. Default is ``'linear'`` (trilinear);
        out-of-bounds coordinates are set to zero.

    Returns
    -------
    torch.Tensor
        Interpolated field, shape ``(*II.shape, C)`` == ``(Rx, Ry, Rz, C)``.

    Raises
    ------
    Exception
        If ``mode`` is not ``'nearest'`` or ``'linear'``.
    """
    num_channels = X.shape[-1]
    if mode == "nearest":
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()

        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = X.shape[0] - 1
        JJr[JJr > (X.shape[1] - 1)] = X.shape[1] - 1
        KKr[KKr > (X.shape[2] - 1)] = X.shape[2] - 1
        Y = torch.zeros([*II.shape, num_channels], device=X.device)
        for channel in range(num_channels):
            Xc = X[..., channel]
            Y[..., channel] = Xc[IIr, JJr, KKr]

    elif mode == "linear":
        #
        ok = (
            (II > 0)
            & (JJ > 0)
            & (KK > 0)
            & (II <= X.shape[0] - 1)
            & (JJ <= X.shape[1] - 1)
            & (KK <= X.shape[2] - 1)
        )
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]
        #
        del JJ, KK
        #
        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = X.shape[0] - 1
        wcx = IIv - fx
        wfx = 1 - wcx
        #
        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = X.shape[1] - 1
        wcy = JJv - fy
        wfy = 1 - wcy
        #
        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = X.shape[2] - 1
        wcz = KKv - fz
        wfz = 1 - wcz
        #
        Y = torch.zeros([*II.shape, num_channels], device=X.device)
        for channel in range(num_channels):
            #
            Xc = X[..., channel]
            #
            c000 = Xc[fx, fy, fz]
            c100 = Xc[cx, fy, fz]
            c010 = Xc[fx, cy, fz]
            c110 = Xc[cx, cy, fz]
            c001 = Xc[fx, fy, cz]
            c101 = Xc[cx, fy, cz]
            c011 = Xc[fx, cy, cz]
            c111 = Xc[cx, cy, cz]
            #
            c00 = c000 * wfx + c100 * wcx
            c01 = c001 * wfx + c101 * wcx
            c10 = c010 * wfx + c110 * wcx
            c11 = c011 * wfx + c111 * wcx
            #
            c0 = c00 * wfy + c10 * wcy
            c1 = c01 * wfy + c11 * wcy
            #
            c = c0 * wfz + c1 * wcz
            #
            Yc = torch.zeros(II.shape, device=X.device)
            Yc[ok] = c.float()
            #
            Y[..., channel] = Yc
        #
    else:
        raise Exception("mode must be linear or nearest")

    return Y


def vol_resample_fast(
    ref_proxy: Nifti1Image,
    flo_proxy: Nifti1Image,
    proxyflow: Optional[Nifti1Image] = None,
    mode: str = "linear",
    device: str = "cpu",
    return_np: bool = False,
) -> Union[np.ndarray, Nifti1Image]:
    """Resample a floating image into the space of a reference image using PyTorch.

    Optionally applies a dense displacement field before resampling.

    Parameters
    ----------
    ref_proxy : nibabel.Nifti1Image
        Reference image defining the output grid.
    flo_proxy : nibabel.Nifti1Image
        Floating (moving) image to resample.
    proxyflow : nibabel.Nifti1Image, optional
        Dense displacement field in the reference space,
        shape ``(3, X, Y, Z)`` or ``(X, Y, Z, 3)``. If ``None``,
        only the affine alignment is applied.
    mode : {'linear', 'nearest'}, optional
        Interpolation mode. Default is ``'linear'``.
    device : str, optional
        PyTorch device string. Default is ``'cpu'``.
    return_np : bool, optional
        If ``True``, return a NumPy array. If ``False`` (default), wrap the
        result in a ``Nifti1Image`` with the reference affine.

    Returns
    -------
    np.ndarray or nibabel.Nifti1Image
        Resampled image.
    """
    ref_v2r = (ref_proxy.affine).astype("float32")
    target_v2r = (flo_proxy.affine).astype("float32")

    ii = np.arange(0, ref_proxy.shape[0], dtype="int32")
    jj = np.arange(0, ref_proxy.shape[1], dtype="int32")
    kk = np.arange(0, ref_proxy.shape[2], dtype="int32")

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing="ij")

    del ii, jj, kk

    II = torch.tensor(II, device=device)
    JJ = torch.tensor(JJ, device=device)
    KK = torch.tensor(KK, device=device)

    if proxyflow is not None:

        flow_v2r = proxyflow.affine
        flow_v2r = flow_v2r.astype("float32")

        affine = torch.tensor(np.linalg.inv(flow_v2r) @ ref_v2r, device=device)
        II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        flow = np.array(proxyflow.dataobj)
        if flow.shape[0] == 3:
            flow = np.transpose(flow, axes=(1, 2, 3, 0))
        flow = torch.tensor(flow, device=device)

        FIELD = fast_3D_interp_field_torch(flow, II2, JJ2, KK2)
        II3 = II2 + FIELD[:, :, :, 0]
        JJ3 = JJ2 + FIELD[:, :, :, 1]
        KK3 = KK2 + FIELD[:, :, :, 2]

        affine = torch.tensor(np.linalg.inv(target_v2r) @ flow_v2r, device=device)
        II4 = (
            affine[0, 0] * II3 + affine[0, 1] * JJ3 + affine[0, 2] * KK3 + affine[0, 3]
        )
        JJ4 = (
            affine[1, 0] * II3 + affine[1, 1] * JJ3 + affine[1, 2] * KK3 + affine[1, 3]
        )
        KK4 = (
            affine[2, 0] * II3 + affine[2, 1] * JJ3 + affine[2, 2] * KK3 + affine[2, 3]
        )

    else:
        affine = torch.tensor(np.linalg.inv(target_v2r) @ ref_v2r, device=device)
        II4 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ4 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK4 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

    image = np.array(flo_proxy.dataobj)
    if len(flo_proxy.shape) == 3:
        reg_image = fast_3D_interp_torch(
            torch.tensor(image, device=device), II4, JJ4, KK4, mode=mode
        )
    else:
        reg_image = fast_3D_interp_field_torch(
            torch.tensor(image, device=device), II4, JJ4, KK4, mode=mode
        )

    reg_image = reg_image.cpu().numpy()

    if return_np:
        return reg_image
    else:
        return nib.Nifti1Image(reg_image, ref_proxy.affine)


def compute_gradient(flow: np.ndarray) -> np.ndarray:
    """Compute the spatial gradient of a 3D vector field using central differences.

    Boundary voxels are left at zero (no padding is applied).

    Parameters
    ----------
    flow : np.ndarray
        Vector field, shape ``(X, Y, Z, 3)``.

    Returns
    -------
    np.ndarray
        Gradient tensor, shape ``(X, Y, Z, 3, 3)``, where entry
        ``[..., i, j]`` is ``d flow_i / d x_j``.
    """
    gradient_map = np.zeros(flow.shape[:3] + (3, 3))

    for it_dim in range(3):
        fmap = flow[..., it_dim]

        dx = fmap[2:, :, :] - fmap[:-2, :, :]
        dy = fmap[:, 2:, :] - fmap[:, :-2, :]
        dz = fmap[:, :, 2:] - fmap[:, :, :-2]

        gradient_map[1:-1, :, :, it_dim, 0] = dx / 2
        gradient_map[:, 1:-1, :, it_dim, 1] = dy / 2
        gradient_map[:, :, 1:-1, it_dim, 2] = dz / 2

    return gradient_map


def compute_jacobian(flow: np.ndarray) -> np.ndarray:
    """Compute the Jacobian determinant of a displacement field.

    Parameters
    ----------
    flow : np.ndarray
        Displacement field, shape ``(X, Y, Z, 3)``.

    Returns
    -------
    np.ndarray
        Jacobian determinant map, shape ``(X, Y, Z)``.  Values < 0 indicate
        folding.
    """
    gradient_map = compute_gradient(flow)
    gradient_map[..., 0, 0] += 1
    gradient_map[..., 1, 1] += 1
    gradient_map[..., 2, 2] += 1
    return np.linalg.det(gradient_map)


def lie_bracket(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the Lie bracket (commutator) of two vector fields.

    The Lie bracket is defined as ``[v, w] = Jw * v - Jv * w``, where
    ``Jv`` and ``Jw`` are the Jacobian matrices of ``v`` and ``w``.

    Parameters
    ----------
    v : np.ndarray
        First vector field, shape ``(X, Y, Z, 3)``.
    w : np.ndarray
        Second vector field, shape ``(X, Y, Z, 3)``.

    Returns
    -------
    np.ndarray
        Lie bracket ``[v, w]``, shape ``(X, Y, Z, 3)``.
    """
    Jv = compute_gradient(v)
    Jw = compute_gradient(w)

    vw = np.einsum("ijklm,ijkm->ijkl", Jw, v) - np.einsum("ijklm,ijkm->ijkl", Jv, w)
    # vw = Jw[..., 0, :]*v[..., 0:1] + Jw[..., 1, :]*v[..., 1:2] + Jw[..., 2, :]*v[..., 2:3] - Jv[..., 0, :]*w[..., 0:1] - Jv[..., 1, :]*w[..., 1:2] - Jv[..., 2, :]*w[..., 2:3]#
    # vw = Jv[..., 0, 0]*w[..., 0] + Jv[..., 1, 1]*w[..., 1] + Jv[..., 2,1]*w[..., 2] - Jw[..., 0,0]*v[..., 0] - Jw[..., 1,1]*v[..., 1] - Jw[..., 2,2]*v[..., 2]#
    # vw = Jw[..., 0]*v[..., 0:1] + Jw[..., 1]*v[..., 1:2] + Jw[..., 2]*v[..., 2:3] - Jv[..., 0]*w[..., 0:1] - Jv[..., 1]*w[..., 1:2] - Jv[..., 2]*w[..., 2:3]#
    return vw  #


def pole_ladder(
    long_svf: np.ndarray, mni_svf: np.ndarray, steps: int = 80
) -> np.ndarray:
    """Parallel-transport a longitudinal SVF along an inter-subject SVF using the pole-ladder scheme.

    Approximates parallel transport on the diffeomorphism group by
    iterating first- and second-order Lie-bracket corrections.

    Parameters
    ----------
    long_svf : np.ndarray
        Longitudinal stationary velocity field to transport,
        shape ``(X, Y, Z, 3)``.
    mni_svf : np.ndarray
        Inter-subject SVF defining the transport path (e.g. subject-to-MNI),
        shape ``(X, Y, Z, 3)``.
    steps : int, optional
        Number of pole-ladder steps. More steps give a better approximation.
        Default is 80.

    Returns
    -------
    np.ndarray
        Transported SVF, shape ``(X, Y, Z, 3)``.
    """
    init_mni_svf = mni_svf / steps
    u = long_svf
    for it_step in range(steps):
        first_term = u
        second_term = lie_bracket(init_mni_svf, u)
        third_term = lie_bracket(init_mni_svf, second_term)
        u = first_term + second_term + 0.5 * third_term

    return u


def svf_to_vox(proxysvf: Nifti1Image) -> Nifti1Image:
    """Convert an SVF from RAS to voxel displacement units.

    Parameters
    ----------
    proxysvf : nibabel.Nifti1Image
        SVF image with displacements in RAS mm, shape ``(X, Y, Z, 3)``.

    Returns
    -------
    nibabel.Nifti1Image
        SVF in voxel units, same affine as ``proxysvf``.
    """
    svf_ras = np.array(proxysvf.dataobj)
    ref_shape = proxysvf.shape[:3]
    svf_ras_zeros = np.concatenate(
        (svf_ras, np.zeros(ref_shape + (1,))), axis=-1
    ).reshape(-1, 4)
    svf_vox = np.dot(np.linalg.inv(proxysvf.affine), svf_ras_zeros.T)
    svf_vox = svf_vox.reshape((4,) + ref_shape)[:3]

    return nib.Nifti1Image(np.transpose(svf_vox, axes=(1, 2, 3, 0)), proxysvf.affine)


def svf_to_ras(proxysvf: Nifti1Image) -> Nifti1Image:
    """Convert an SVF from voxel to RAS displacement units.

    Equivalent to multiplying the voxel SVF by the voxel-to-RAS matrix with
    a zero translation column (the mosaic centering terms cancel out).

    Parameters
    ----------
    proxysvf : nibabel.Nifti1Image
        SVF image with displacements in voxel units, shape ``(X, Y, Z, 3)``.

    Returns
    -------
    nibabel.Nifti1Image
        SVF in RAS mm, same affine as ``proxysvf``.
    """
    svf_vox = np.array(proxysvf.dataobj)
    ref_shape = proxysvf.shape[:3]
    svf_vox_zeros = np.concatenate(
        (svf_vox, np.zeros(ref_shape + (1,))), axis=-1
    ).reshape(-1, 4)
    svf_ras = np.dot(proxysvf.affine, svf_vox_zeros.T)
    svf_ras = svf_ras.reshape((4,) + ref_shape)[:3]

    # II, JJ, KK = np.meshgrid(np.arange(0, ref_shape[0]), np.arange(0, ref_shape[1]), np.arange(0, ref_shape[2]), indexing='ij')
    #
    # def_vox = np.zeros_like(svf_vox)
    # def_vox[..., 0] = svf_vox[..., 0] + II
    # def_vox[..., 1] = svf_vox[..., 1] + JJ
    # def_vox[..., 2] = svf_vox[..., 2] + KK
    #
    # def_vox_ones = np.concatenate((def_vox, np.ones(ref_shape + (1,))), axis=-1).reshape(-1, 4)
    # ref_vox_ones = np.concatenate((II[..., np.newaxis], JJ[..., np.newaxis], KK[..., np.newaxis], np.ones(ref_shape + (1,))), axis=-1).reshape(-1, 4)
    #
    # def_ras = np.dot(proxysvf.affine, def_vox_ones.T)
    # ref_ras = np.dot(proxysvf.affine, ref_vox_ones.T)
    # svf_ras = def_ras-ref_ras
    # svf_ras = svf_ras.reshape((4,) + ref_shape)[:3]

    return nib.Nifti1Image(np.transpose(svf_ras, axes=(1, 2, 3, 0)), proxysvf.affine)


def network_space(
    im: sf.Volume, shape: Sequence[int], center: Optional[sf.Volume] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct transform from network space to the voxel space of an image.

    Constructs a coordinate transform from the space the network will operate
    in to the zero-based image index space. The network space has isotropic
    1-mm voxels, left-inferior-anterior (LIA) orientation, and no shear. It is
    centered on the field of view, or that of a reference image. This space is
    an indexed voxel space, not world space.

    Parameters
    ----------
    im : surfa.Volume
        Input image to construct the transform for.
    shape : (3,) array-like
        Spatial shape of the network space.
    center : surfa.Volume, optional
        Center the network space on the center of a reference image.

    Returns
    -------
    out : tuple of (3, 4) NumPy arrays
        Transform from network to input-image space and its inverse, thinking
        coordinates.

    """
    old = im.geom
    new = sf.ImageGeometry(
        shape=shape,
        voxsize=1,
        rotation="LIA",
        center=old.center if center is None else center.geom.center,
        shear=None,
    )

    net_to_vox = old.world2vox @ new.vox2world
    vox_to_net = new.world2vox @ old.vox2world
    return (
        np.float32(net_to_vox.matrix),
        np.float32(vox_to_net.matrix),
        new.vox2world.matrix,
    )


def getM(ref: np.ndarray, mov: np.ndarray, use_L1: bool = False) -> np.ndarray:
    """Estimate a 4×4 affine matrix from point correspondences.

    Solves the least-squares problem ``mov ≈ M * ref`` for a 12-DOF affine
    matrix. Optionally refines using L1 minimisation via linear programming.

    Parameters
    ----------
    ref : np.ndarray
        Reference points, shape ``(3, N)``.
    mov : np.ndarray
        Corresponding moving points, shape ``(3, N)``.
    use_L1 : bool, optional
        If ``True``, refine the L2 solution with an L1 minimisation step.
        Default is ``False``.

    Returns
    -------
    np.ndarray
        Estimated affine matrix, shape ``(4, 4)``.
    """
    zmat = np.zeros(ref.shape[::-1])
    zcol = np.zeros([ref.shape[1], 1])
    ocol = np.ones([ref.shape[1], 1])
    zero = np.zeros(zmat.shape)

    A = np.concatenate(
        [
            np.concatenate([np.transpose(ref), zero, zero, ocol, zcol, zcol], axis=1),
            np.concatenate([zero, np.transpose(ref), zero, zcol, ocol, zcol], axis=1),
            np.concatenate([zero, zero, np.transpose(ref), zcol, zcol, ocol], axis=1),
        ],
        axis=0,
    )

    b = np.concatenate(
        [np.transpose(mov[0, :]), np.transpose(mov[1, :]), np.transpose(mov[2, :])],
        axis=0,
    )

    x = np.matmul(
        np.linalg.inv(np.matmul(np.transpose(A), A)), np.matmul(np.transpose(A), b)
    )

    # If L1 minimization: we use L2 solution as initialization and miminize L1
    if use_L1:
        Apos = np.concatenate([A, -np.eye(A.shape[0])], axis=1)
        Aneg = np.concatenate([-A, -np.eye(A.shape[0])], axis=1)
        A_ub = np.concatenate([Apos, Aneg], axis=0)
        b_ub = np.concatenate([b, -b])
        c = np.concatenate([np.zeros(12), np.ones(A.shape[0])])

        aux = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            method="interior-point",
            bounds=[None, None],
            options={"disp": True, "autoscale": True},
            x0=np.concatenate([x, 0.1 + np.abs(np.matmul(A, x) - b)]),
        )
        x = aux.x[0:12]

    M = np.stack(
        [
            [x[0], x[1], x[2], x[9]],
            [x[3], x[4], x[5], x[10]],
            [x[6], x[7], x[8], x[11]],
            [0, 0, 0, 1],
        ]
    )

    return M


def create_empty_template(
    image_list: list[str | Nifti1Image], margin_bb: int = 5
) -> tuple[np.ndarray, np.ndarray, tuple[int]]:
    """Build a bounding-box template that encompasses a list of brain images.

    Computes the average RAS bounding box across all images (with a 5 mm
    margin) and returns a 1 mm isotropic identity-oriented template grid.

    Parameters
    ----------
    image_list : list of str or nibabel.Nifti1Image
        Images whose non-zero voxel extents define the bounding box.
    margin_bb : int
        Margin from each axis from the mask

    Returns
    -------
    ras_mosaic : np.ndarray
        RAS coordinates of all template voxels, shape ``(4, N_vox)``.
    template_vox2ras0 : np.ndarray
        Voxel-to-RAS affine of the template, shape ``(4, 4)``.
    template_size : tuple of int
        Spatial dimensions ``(X, Y, Z)`` of the template grid.
    """
    boundaries_min = np.zeros((len(image_list), 3))
    boundaries_max = np.zeros((len(image_list), 3))
    for it_lil, lil in enumerate(image_list):

        if isinstance(lil, nib.nifti1.Nifti1Image):
            proxy = lil
        else:
            proxy = nib.load(lil)

        mask = np.asarray(proxy.dataobj)
        header = proxy.affine
        idx = np.where(mask > 0)
        vox_min = np.concatenate((np.min(idx, axis=1), [1]), axis=0)
        vox_max = np.concatenate((np.max(idx, axis=1), [1]), axis=0)

        minR, minA, minS = np.inf, np.inf, np.inf
        maxR, maxA, maxS = -np.inf, -np.inf, -np.inf

        for i in [vox_min[0], vox_max[0] + 1]:
            for j in [vox_min[1], vox_max[1] + 1]:
                for k in [vox_min[2], vox_max[2] + 1]:
                    aux = np.dot(header, np.asarray([i, j, k, 1]).T)

                    minR, maxR = min(minR, aux[0]), max(maxR, aux[0])
                    minA, maxA = min(minA, aux[1]), max(maxA, aux[1])
                    minS, maxS = min(minS, aux[2]), max(maxS, aux[2])

        minR -= margin_bb
        minA -= margin_bb
        minS -= margin_bb

        maxR += margin_bb
        maxA += margin_bb
        maxS += margin_bb

        boundaries_min[it_lil] = [minR, minA, minS]
        boundaries_max[it_lil] = [maxR, maxA, maxS]
        # boundaries_min += [[minR, minA, minS]]
        # boundaries_max += [[maxR, maxA, maxS]]

    # Get the corners of cuboid in RAS space
    minR = np.mean(boundaries_min[..., 0])
    minA = np.mean(boundaries_min[..., 1])
    minS = np.mean(boundaries_min[..., 2])
    maxR = np.mean(boundaries_max[..., 0])
    maxA = np.mean(boundaries_max[..., 1])
    maxS = np.mean(boundaries_max[..., 2])

    template_size = np.asarray(
        [
            int(np.ceil(maxR - minR)) + 1,
            int(np.ceil(maxA - minA)) + 1,
            int(np.ceil(maxS - minS)) + 1,
        ]
    )

    # Define header and size
    template_vox2ras0 = np.asarray(
        [[1, 0, 0, minR], [0, 1, 0, minA], [0, 0, 1, minS], [0, 0, 0, 1]]
    )

    # VOX Mosaic
    II, JJ, KK = np.meshgrid(
        np.arange(0, template_size[0]),
        np.arange(0, template_size[1]),
        np.arange(0, template_size[2]),
        indexing="ij",
    )

    RR = II + minR
    AA = JJ + minA
    SS = KK + minS
    ras_mosaic = np.concatenate(
        (
            RR.reshape(-1, 1),
            AA.reshape(-1, 1),
            SS.reshape(-1, 1),
            np.ones((np.prod(template_size), 1)),
        ),
        axis=1,
    ).T

    return ras_mosaic, template_vox2ras0, tuple(template_size)


class VecInt(nn.Module):
    """Integrate a stationary velocity field via scaling and squaring.

    Implements the scaling-and-squaring algorithm for diffeomorphic
    registration. If you find this layer useful, please cite:

        Dalca et al., *Unsupervised Learning for Fast Probabilistic
        Diffeomorphic Registration*, MICCAI 2018.

    Attributes
    ----------
    int_steps : int
        Number of integration steps.
    scale : float
        Initial scaling factor ``1 / 2**int_steps``.
    transformer : SpatialTransformer
        Spatial transformer used to compose intermediate deformations.
    """

    def __init__(
        self, field_shape: tuple[int, ...], int_steps: int = 7, **kwargs
    ) -> None:
        """
        Parameters
        ----------
        field_shape : tuple of int
            Spatial shape of the velocity field ``(X, Y, Z)``.
        int_steps : int, optional
            Number of scaling-and-squaring steps. Default is 7.
        """
        super().__init__()
        self.int_steps = int_steps
        self.scale = 1 / (2**self.int_steps)
        self.transformer = SpatialTransformer(field_shape)

    def forward(self, field: torch.Tensor, **kwargs) -> torch.Tensor:
        """Integrate the velocity field.

        Parameters
        ----------
        field : torch.Tensor
            Stationary velocity field, shape ``(B, 3, X, Y, Z)``.
        **kwargs
            Optional ``nsteps`` override to reduce integration steps at
            inference.

        Returns
        -------
        torch.Tensor
            Integrated displacement field, same shape as ``field``.
        """

        output = field
        output = output * self.scale
        nsteps = self.int_steps
        if "nsteps" in kwargs:
            nsteps = nsteps - kwargs["nsteps"]

        for _ in range(nsteps):
            a = self.transformer(output, output)
            output = output + a

        return output


class RescaleTransform(nn.Module):
    """Resize a displacement field and rescale its vector magnitudes accordingly.

    Resampling a flow field requires both interpolating the field to a new
    grid *and* multiplying vector components by the corresponding scale
    factors. Optionally applies Gaussian anti-aliasing before downsampling.

    Credit: VoxelMorph (https://github.com/voxelmorph/voxelmorph).

    Attributes
    ----------
    ndims : int
        Number of spatial dimensions.
    factor : list of float
        Per-dimension scale factors.
    """

    def __init__(
        self,
        inshape: tuple[int, ...],
        factor: Optional[Union[float, list[float], tuple[float, ...]]] = None,
        target_size: Optional[tuple[int, ...]] = None,
        gaussian_filter_flag: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        inshape : tuple of int
            Spatial shape of the input field.
        factor : float or list of float, optional
            Isotropic or per-dimension scale factor. Mutually exclusive with
            ``target_size``.
        target_size : tuple of int, optional
            Target spatial shape from which the scale factor is derived.
            Used only when ``factor`` is ``None``.
        gaussian_filter_flag : bool, optional
            If ``True``, apply Gaussian anti-aliasing before downsampling.
            Default is ``False``.
        """
        super().__init__()

        self.ndims = len(inshape)
        self.mode = "linear"
        self.inshape = inshape
        self.gaussian_filter_flag = gaussian_filter_flag

        if factor is None:
            assert target_size is not None
            self.factor = tuple([b / a for a, b in zip(inshape, target_size)])
        elif isinstance(factor, list) or isinstance(factor, tuple):
            self.factor = list(factor)
        else:
            self.factor = [factor for _ in range(self.ndims)]

        if self.ndims == 2:
            self.mode = "bi" + self.mode
        elif self.ndims == 3:
            self.mode = "tri" + self.mode

        if self.factor[0] < 1 and self.gaussian_filter_flag:
            kernel_sigma = [0.44 * 1 / f for f in self.factor]

            if self.ndims == 2:
                kernel = self.gaussian_filter_2d(kernel_sigma=kernel_sigma)
            elif self.ndims == 3:
                kernel = self.gaussian_filter_3d(kernel_sigma=kernel_sigma)
            else:
                raise ValueError("[RESCALE TF] No valid kernel found.")
            self.register_buffer("kernel", kernel)

    def gaussian_filter_2d(
        self, kernel_sigma: Union[float, list[float]]
    ) -> torch.Tensor:
        """Build a 2D separable Gaussian convolution kernel as a registered buffer.

        Parameters
        ----------
        kernel_sigma : float or list of float
            Standard deviation(s) of the Gaussian in each spatial dimension.

        Returns
        -------
        torch.Tensor
            Kernel tensor of shape ``(2, 2, ky, kx)``.
        """
        if isinstance(kernel_sigma, list):
            kernel_size = [
                int(np.ceil(ks * 3) + np.mod(np.ceil(ks * 3) + 1, 2))
                for ks in kernel_sigma
            ]

        else:
            kernel_size = int(
                np.ceil(kernel_sigma * 3) + np.mod(np.ceil(kernel_sigma * 3) + 1, 2)
            )

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX = np.meshgrid(coord[0], coord[1], indexing="ij")
        xy_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2.0 for ks in kernel_size])
        mean = mean.reshape(-1, 1, 1)
        variance = np.asarray([ks**2.0 for ks in kernel_sigma])
        variance = variance.reshape(-1, 1, 1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)

        norm_kernel = 1.0 / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma))
        kernel = norm_kernel * np.exp(
            -np.sum((xy_grid - mean) ** 2.0 / (2 * variance), axis=0)
        )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1])

        # Total kernel
        total_kernel = np.zeros((2, 2) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel

        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def gaussian_filter_3d(
        self, kernel_sigma: Union[float, list[float]]
    ) -> torch.Tensor:
        """Build a 3D separable Gaussian convolution kernel as a registered buffer.

        Parameters
        ----------
        kernel_sigma : float or list of float
            Standard deviation(s) of the Gaussian in each spatial dimension.

        Returns
        -------
        torch.Tensor
            Kernel tensor of shape ``(3, 3, kz, ky, kx)``.
        """
        if isinstance(kernel_sigma, list):
            kernel_size = [
                int(np.ceil(ks * 3) + np.mod(np.ceil(ks * 3) + 1, 2))
                for ks in kernel_sigma
            ]

        else:
            kernel_size = int(
                np.ceil(kernel_sigma * 3) + np.mod(np.ceil(kernel_sigma * 3) + 1, 2)
            )

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX, ZZ = np.meshgrid(coord[0], coord[1], coord[2], indexing="ij")
        xyz_grid = np.concatenate(
            (YY[np.newaxis], XX[np.newaxis], ZZ[np.newaxis]), axis=0
        )  # 2, y, x

        mean = np.asarray([(ks - 1) / 2.0 for ks in kernel_size])
        mean = mean.reshape(-1, 1, 1, 1)
        variance = np.asarray([ks**2.0 for ks in kernel_sigma])
        variance = variance.reshape(-1, 1, 1, 1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)
        norm_kernel = 1.0 / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma))
        kernel = norm_kernel * np.exp(
            -np.sum((xyz_grid - mean) ** 2.0 / (2 * variance), axis=0)
        )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1], kernel_size[2])

        # Total kernel

        total_kernel = np.zeros((3, 3) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel
        total_kernel[2, 2] = kernel

        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale the displacement field.

        Parameters
        ----------
        x : torch.Tensor
            Input flow field, shape ``(B, ndims, *spatial)``.

        Returns
        -------
        torch.Tensor
            Rescaled flow field at the new spatial resolution.
        """
        x = x.clone()
        if self.factor[0] < 1:
            if self.gaussian_filter_flag:
                padding = [int((s - 1) // 2) for s in self.kernel.shape[2:]]
                if self.ndims == 2:
                    x = F.conv2d(x, self.kernel, stride=(1, 1), padding=padding)
                else:
                    x = F.conv3d(x, self.kernel, stride=(1, 1, 1), padding=padding)

            # resize first to save memory
            x = F.interpolate(
                x, align_corners=True, scale_factor=self.factor, mode=self.mode
            )
            for i in range(self.ndims):
                x[:, i] = x[:, i] * self.factor[i]

        elif self.factor[0] > 1:
            # multiply first to save memory
            for i in range(self.ndims):
                x[:, i] = x[:, i] * self.factor[i]
            x = F.interpolate(
                x, align_corners=True, scale_factor=self.factor, mode=self.mode
            )

        # don't do anything if resize is 1
        return x


class SpatialInterpolation(nn.Module):
    """Apply a pre-computed normalised grid to a source image via ``grid_sample``.

    Unlike :class:`SpatialTransformer`, this layer takes an explicit
    normalised location grid rather than a flow field relative to a
    pre-built identity grid.

    Credit: VoxelMorph
    (https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py).
    """

    def __init__(self, mode: str = "bilinear", padding_mode: str = "zeros") -> None:
        """
        Parameters
        ----------
        mode : str, optional
            Interpolation mode passed to ``F.grid_sample``. Default is
            ``'bilinear'``.
        padding_mode : str, optional
            Padding mode for out-of-boundary coordinates. Default is
            ``'zeros'``.
        """
        super().__init__()

        self.mode = mode
        self.padding_mode = padding_mode

    def forward(
        self, src: torch.Tensor, new_locs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Interpolate ``src`` at the given normalised coordinates.

        Parameters
        ----------
        src : torch.Tensor
            Source image, shape ``(B, C, *spatial)``.
        new_locs : torch.Tensor
            Sampling grid in voxel coordinates, shape
            ``(B, ndims, *spatial)``. Will be normalised to ``[-1, 1]``
            internally.
        **kwargs
            Optional ``'padding_mode'`` and ``'mode'`` overrides.

        Returns
        -------
        torch.Tensor
            Warped image, same shape as ``src``.
        """
        if "padding_mode" in kwargs:
            self.padding_mode = kwargs["padding_mode"]
        if "mode" in kwargs:
            self.mode = kwargs["mode"]

        shape = src.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(
            src,
            new_locs,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )


class SpatialTransformer(nn.Module):
    """Warp a source image by adding a flow field to an identity grid.

    Registers a fixed identity sampling grid at construction time and adds
    the flow field to it at forward time before calling ``F.grid_sample``.

    Credit: VoxelMorph
    (https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py).

    Attributes
    ----------
    grid : torch.Tensor
        Pre-computed identity sampling grid, shape ``(1, ndims, *size)``.
    """

    def __init__(
        self,
        size: tuple[int, ...],
        mode: str = "bilinear",
        padding_mode: str = "border",
    ) -> None:
        """
        Parameters
        ----------
        size : tuple of int
            Spatial shape of the images to transform, e.g. ``(X, Y, Z)``.
        mode : str, optional
            Interpolation mode for ``F.grid_sample``. Default is
            ``'bilinear'``.
        padding_mode : str, optional
            Padding mode for out-of-boundary coordinates. Default is
            ``'border'``.
        """
        super().__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, src: torch.Tensor, flow: torch.Tensor, **kwargs) -> torch.Tensor:
        """Warp ``src`` by adding ``flow`` to the pre-built identity grid.

        Parameters
        ----------
        src : torch.Tensor
            Source image, shape ``(B, C, *spatial)``.
        flow : torch.Tensor
            Displacement field, shape ``(B, ndims, *spatial)``.
        **kwargs
            Optional ``'padding_mode'`` and ``'mode'`` overrides.

        Returns
        -------
        torch.Tensor
            Warped image, same shape as ``src``.
        """
        padding_mode = (
            kwargs["padding_mode"] if "padding_mode" in kwargs else self.padding_mode
        )
        mode = kwargs["mode"] if "mode" in kwargs else self.mode

        new_locs = self.grid + flow
        shape = src.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            # new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(
            src, new_locs, mode=mode, padding_mode=padding_mode, align_corners=True
        )



def register_to_MNI(
    im_filepath: str | Path,
    seg_filepath: str | Path,
    MNI_TEMPLATE_SEG: str | Path,
    MNI_TEMPLATE: str | Path,
    labels_registration: np.ndarray,
) -> Tuple[nib.Nifti1Image, np.ndarray]:
    """
    Register a subject image to MNI space using centroid-based affine alignment.

    Computes an affine transform by matching label centroids between the subject
    segmentation and the MNI template segmentation, applies optional Gaussian
    smoothing to account for resolution differences, resamples the image into
    MNI space, and saves the result.

    Parameters
    ----------
    im_filepath : str or Path
        Path to the subject's input image (.nii.gz).
    seg_filepath : str or Path
        Path to the subject's SynthSeg segmentation file (.nii.gz), used for centroid
        computation.
    MNI_TEMPLATE_SEG : str or Path
        Path to the MNI template segmentation, used as the registration reference
        for centroid computation.
    MNI_TEMPLATE : str or Path
        Path to the MNI template image, used to define the target voxel grid and
        resolution for resampling.
    labels_registration : array-like
        List or array of label indices to use for centroid matching between the
        subject and MNI segmentations.

    Returns
    -------
    None
        Saves two files to disk:
        - Registered image at `save_path`
        - Affine matrix at `save_path.replace('.nii.gz', 'aff.npy')`

    """
    centroid_ref, ok_ref = compute_centroids_ras(
        MNI_TEMPLATE_SEG, labels_registration
    )
    centroid_flo, ok_flo = compute_centroids_ras(
        seg_filepath, labels_registration
    )

    aff_MNI = getM(
        centroid_ref[:, ok_ref > 0], centroid_flo[:, ok_ref > 0], use_L1=False
    )
    # np.save(save_path.replace('.nii.gz', 'aff.npy'), aff_MNI)

    mni_proxy = nib.load(MNI_TEMPLATE)
    im_proxy = nib.load(im_filepath)
    voxsize = np.sqrt(np.sum(im_proxy.affine * im_proxy.affine, axis=0))[:-1]
    voxsize_new = np.sqrt(np.sum(mni_proxy.affine * mni_proxy.affine, axis=0))[:-1]
    factor = voxsize / voxsize_new
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    im_array = np.array(im_proxy.dataobj)
    im_array = gaussian_filter(im_array, sigmas)
    im_proxy = nib.Nifti1Image(im_array, np.linalg.inv(aff_MNI) @ im_proxy.affine)
    im_proxy = vol_resample_fast(mni_proxy, im_proxy, return_np=False)
    # nib.save(im_proxy, save_path)

    return im_proxy, aff_MNI