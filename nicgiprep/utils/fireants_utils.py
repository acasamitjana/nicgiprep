"""
FireANTs-based replacement for the TensorFlow/SynthMorph deformable-registration utilities.

Provides ``fireants_register`` and ``integrate_svf`` as drop-in replacements for
``nicgiprep.utils.synthmorph_utils.synthmorph_register`` and ``...integrate_svf``: both
functions preserve the original array contracts (a forward stationary velocity field (SVF),
and a scaling-and-squaring integration of an SVF into a dense displacement field) so that
callers in ``nicgiprep.pipelines`` do not need to change beyond the import.

Coordinate conventions
-----------------------
FireANTs represents images and velocity fields in a normalised ``[-1, 1]`` torch grid
space (see ``fireants.io.image.Image``), whereas the rest of nicgiprep represents
deformation fields as voxel-index displacements tied to a nibabel affine (see
``nicgiprep.utils.def_utils``). ``fireants_register`` converts between the two using the
``Image.torch2px`` transform, which FireANTs itself builds as a per-axis diagonal scale
(independent of the image's spacing/direction/origin). Both the reference and floating
image are resampled onto the exact same target voxel grid (``svf_shape``/``svf_v2r``)
before being handed to FireANTs, so the pixel index of that resampled grid is exactly the
voxel index of ``svf_v2r`` by construction, and the physical (spacing/direction/origin)
geometry we assign only needs to be self-consistent between the two images, not
independently correct -- it does not affect the returned SVF's voxel-displacement values.
"""
from typing import Optional, Union, Sequence

import numpy as np
import nibabel as nib
import torch
import SimpleITK as sitk
from bids.layout import BIDSFile
from torch.nn import functional as F

from fireants.io.image import Image as FAImage, BatchedImages
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.deformation.svf import StationaryVelocity
from fireants.losses.cc import gaussian_1d, separable_filtering

#: nibabel/nicgiprep affines are voxel-to-RAS; SimpleITK expects voxel-to-LPS.
_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def _resolve_device(device: Optional[str]) -> str:
    """Return the requested device, defaulting to CUDA when available.

    Parameters
    ----------
    device : str, optional
        Explicit PyTorch device string. If ``None``, resolved at call time.

    Returns
    -------
    str
        ``device`` unchanged, or ``'cuda'``/``'cpu'`` depending on availability.
    """
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _affine_to_sitk_geometry(affine: np.ndarray) -> tuple:
    """Decompose a nibabel (RAS) voxel-to-world affine into sitk (LPS) geometry.

    Parameters
    ----------
    affine : np.ndarray
        4x4 voxel-to-RAS affine.

    Returns
    -------
    spacing : tuple of float
        Per-axis voxel spacing.
    direction : tuple of float
        Flattened 3x3 direction cosine matrix (row-major, as expected by
        ``SimpleITK.Image.SetDirection``).
    origin : tuple of float
        Physical (LPS) origin.
    """
    lps_affine = _RAS_TO_LPS @ affine
    axes = lps_affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    direction = axes / spacing
    origin = lps_affine[:3, 3]
    return tuple(spacing.tolist()), tuple(direction.flatten(order="C").tolist()), tuple(origin.tolist())


def _array_to_fireants_image(array: np.ndarray, affine: np.ndarray, device: str) -> FAImage:
    """Wrap a ``(X, Y, Z)`` numpy array and its nibabel affine as a FireANTs ``Image``.

    Parameters
    ----------
    array : np.ndarray
        Image intensities, shape ``(X, Y, Z)``, in nibabel voxel-index order.
    affine : np.ndarray
        4x4 voxel-to-RAS affine associated with ``array``.
    device : str
        PyTorch device string for the resulting ``Image``.

    Returns
    -------
    fireants.io.image.Image
        FireANTs image wrapping ``array`` with matching physical geometry.
    """
    spacing, direction, origin = _affine_to_sitk_geometry(affine)
    # SimpleITK arrays are indexed (Z, Y, X); nibabel arrays are indexed (X, Y, Z).
    itk_image = sitk.GetImageFromArray(np.transpose(array, (2, 1, 0)).astype(np.float32))
    itk_image.SetSpacing(spacing)
    itk_image.SetDirection(direction)
    itk_image.SetOrigin(origin)
    return FAImage(itk_image, device=device)

def _grad_penalty(input_tensor: torch.Tensor) -> torch.Tensor:
    '''
    Calculates the magnitude of the partial gradients in every dimension

    Parameters
    ----------
    input_tensor: torch.Tensor
        deformation/displacement/warp/svf field. It expects 5-D input tensor with number of channels equivalent to
        the number of spatial dimensions

    Returns
    -------
    torch.Tensor
        gradient calculation
    '''
    num_spatial = input_tensor.ndim - 2
    assert num_spatial >= 1, f"Need at least 1 spatial dim to compute gradients, got {num_spatial}"
    gradients = []
    for spatial_dim in range(num_spatial):
        gradients.append(torch.diff(input_tensor, dim=1 + spatial_dim) * 0.5 * 192) #192=typical size of a brain in 1mm3

    penalty = 'l2'
    if penalty == 'l1':
        penalties = [torch.mean(g.abs()) for g in gradients]
    else:
        penalties = [torch.mean(g * g) for g in gradients]

    return sum(penalties) / len(penalties)

def fireants_register(
    imref_file: Union[str, BIDSFile],
    imflo_file: Union[str, BIDSFile],

    cc_kernel_size: int = 5,
    smooth_warp_sigma: float = 0.5,
    smooth_grad_sigma: float = 1.0,
    int_steps: int = 7,
    device: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Register a floating image to a reference using FireANTs geodesic deformable registration.

    Replaces ``nicgiprep.utils.synthmorph_utils.synthmorph_register``. Both images are
    first resampled onto the shared ``svf_shape``/``svf_v2r`` grid -- the same grid every
    pairwise SVF for a subject is defined on in the USLR longitudinal/multimodal pipelines
    -- and FireANTs' ``GreedyRegistration`` with a stationary-velocity (geodesic) deformation
    model is optimised directly at that resolution (``scales=[1]``), so the returned
    velocity field needs no further resampling before being combined across session pairs.

    Parameters
    ----------
    imref_file : str, nibabel.Nifti1Image, or file-like
        Reference (fixed) image. Accepts a file path, a ``Nifti1Image`` object, or an
        object with a ``.path`` attribute.
    imflo_file : str, nibabel.Nifti1Image, or file-like
        Floating (moving) image. Same format options as ``imref_file``.
    svf_shape : tuple of int
        Spatial shape of the shared SVF grid, e.g. ``self.svf_shape``.
    svf_v2r : np.ndarray
        4x4 voxel-to-RAS affine of the shared SVF grid, e.g. loaded from ``self.svf_v2r_ent``.
    iterations : int, optional
        Number of optimisation iterations. Default is 100.
    cc_kernel_size : int, optional
        Local normalised cross-correlation kernel size, in voxels. Default is 5.
    smooth_warp_sigma : float, optional
        Gaussian smoothing sigma applied to the warp field, in voxels. Default is 0.5.
    smooth_grad_sigma : float, optional
        Gaussian smoothing sigma applied to the velocity-field gradient, in voxels.
        Default is 1.0.
    int_steps : int, optional
        Number of scaling-and-squaring steps used internally by FireANTs to integrate the
        velocity field during optimisation. Default is 7.
    device : str, optional
        PyTorch device string. Defaults to ``'cuda'`` if available, else ``'cpu'``.

    Returns
    -------
    np.ndarray or None
        Forward SVF, shape ``svf_shape + (3,)``, in voxel-index-displacement units of the
        ``svf_v2r`` grid. ``None`` if registration failed.
    """
    device = _resolve_device(device)

    if isinstance(imref_file, BIDSFile):
        imref_file = imref_file.path
    if isinstance(imflo_file, BIDSFile):
        imflo_file = imflo_file.path

    fa_ref = FAImage.load_file(imref_file)
    fa_flo = FAImage.load_file(imflo_file)

    scales = [4, 2]
    iterations = [300, 100]

    reg = GreedyRegistration(
        scales=scales,
        iterations=iterations,
        optimizer_lr=2e-3,
        fixed_images=BatchedImages(fa_ref),
        moving_images=BatchedImages(fa_flo),
        loss_type="cc",
        deformation_type="geodesic",
        cc_kernel_size=cc_kernel_size,
        integrator_n=int_steps,
        smooth_warp_sigma=smooth_warp_sigma,
        smooth_grad_sigma=smooth_grad_sigma,
        progress_bar=False,
        displacement_reg=_grad_penalty
    )
    reg.optimize()
    svf = reg.warp.velocity_field.detach()[0]  # [X, Y, Z, 3], normalised torch units
    svf = svf.cpu().numpy()  # -> [X, Y, Z, 3]

    proxysvf = _fireants_field_to_nib(svf, fa_ref, units='vox', index=0)

    return proxysvf


def integrate_svf(
        proxyref,
        proxysvf,
        scaling_factor: int = 2,
        int_steps: int = 7,
        smooth_warp_sigma: float = 1.,
        return_np: bool = True,
        device: Optional[str] = None,
) -> np.ndarray:
    """Integrate a stationary velocity field (SVF) via scaling and squaring.

    Pure-PyTorch drop-in replacement for
    ``nicgiprep.utils.synthmorph_utils.integrate_svf``, built on the ``VecInt``/
    ``RescaleTransform`` layers already available in :mod:`nicgiprep.utils.def_utils`.

    Parameters
    ----------
    svf : np.ndarray
        Stationary velocity field, shape ``(X, Y, Z, 3)``, in voxel-index-displacement
        units of its own grid.
    orig_shape : tuple of int
        Spatial shape of the target image space that the returned flow is resampled onto.
    scaling_factor : int, optional
        Expected upscaling factor between ``svf``'s grid and ``orig_shape`` (``svf.shape[:3]``
        times ``scaling_factor`` should equal ``orig_shape``). Kept for API compatibility;
        the actual resampling factor is derived directly from the two shapes. Default is 2.
    int_steps : int, optional
        Number of scaling-and-squaring steps. Default is 7.
    return_np : bool, optional
        Either return a nibabel object or a tuple (array, affine)
    Returns
    -------
    np.ndarray
        Dense displacement field, shape ``orig_shape + (3,)``, in voxel-index-displacement
        units of the ``orig_shape`` grid.
    """
    fake_ref = BatchedImages(_array_to_fireants_image(np.array(proxyref.dataobj), proxyref.affine, device=device))
    svf = _nib_field_to_fireants(np.array(proxysvf.dataobj), proxysvf.affine, fake_ref, "vox", 0)

    fa_svf = StationaryVelocity(fake_ref, fake_ref, integrator_n=int_steps, init_scale=scaling_factor)
    with torch.no_grad():
        fa_svf.velocity_field.copy_(svf)

    # smooth out the warp field if asked to
    warp_field = fa_svf.get_warp()
    warp_field = F.interpolate(warp_field.permute(*fa_svf.permute_vtoimg), size=fake_ref.shape[2:], mode="trilinear",
                               align_corners=True).permute(*fa_svf.permute_imgtov)
    if smooth_warp_sigma > 0:
        warp_gaussian = [gaussian_1d(s, truncated=2) for s in
                         (torch.zeros(3, device=fake_ref.device, dtype=torch.float32) + smooth_warp_sigma)]
        warp_field = separable_filtering(warp_field.permute(*fa_svf.permute_vtoimg), warp_gaussian).permute(*fa_svf.permute_imgtov)

    proxyflow = _fireants_field_to_nib(warp_field.detach().cpu().numpy()[0].astype(np.float64), fake_ref, "vox", 0)
    if return_np:
        return np.array(proxyflow.dataobj), proxyflow.affine
    else:
        return proxyflow

def _fireants_field_to_nib(
        field: np.ndarray, image, units: str, index: int
) -> nib.Nifti1Image:
    """Convert a FireANTs field in normalized torch coordinates to a nibabel image.

    Shared by :func:`fireants_register` and :func:`integrate_svf`: both
    export a ``[N, z, y, x, 3]`` tensor in **normalized torch coordinates**
    (``[-1, 1]``, with component 0 indexing the last tensor axis, i.e. the ITK
    x axis), so the conversion is: components ``-> LPS mm`` with the linear
    part of ``torch2phy``, then ``-> RAS``, then the ``(z, y, x) -> (x, y, z)``
    transpose.

    Parameters
    ----------
    field : np.ndarray
        Field for a single batch index, shape ``(z, y, x, 3)``, in normalized
        torch coordinates.
    image : fireants.io.image.Image or fireants.io.image.BatchedImages
        The (batched) image whose ``torch2phy`` defines the grid's physical
        geometry -- e.g. the moving or fixed image used to build the
        registration that produced ``field``.
    units : {'ras', 'vox'}
        Units of the components of the returned field. ``'vox'`` is in voxels
        *of the field's own grid*.
    index : int
        Batch index (only used to select the matching ``torch2phy``).

    Returns
    -------
    nibabel.Nifti1Image
        Field of shape ``(X, Y, Z, 3)`` with an RAS affine.
    """
    affine = _fireants_grid_affine(image, field.shape[:3], index=index)

    # components: normalized -> LPS mm -> RAS
    t2p = _fireants_t2p(image, index=index)
    data = field @ t2p[:3, :3].T
    data = data @ _RAS_TO_LPS[:3, :3].T

    # (z, y, x, 3) -> (x, y, z, 3)
    data = np.transpose(data, (2, 1, 0, 3))

    return nib.Nifti1Image(_to_units(data, affine, units), affine)


def _nib_field_to_fireants(
        data: np.ndarray, affine: np.ndarray, image, units: str, index: int
) -> torch.Tensor:
    """Invert :func:`_fireants_field_to_nib`: nibabel field -> FireANTs normalized tensor.

    Parameters
    ----------
    data : np.ndarray
        Field values, shape ``(X, Y, Z, 3)``, in ``units`` and in nibabel
        voxel-index order (matching ``affine``).
    affine : np.ndarray
        4x4 voxel-to-RAS affine of ``data``'s grid.
    image : fireants.io.image.Image or fireants.io.image.BatchedImages
        The image whose ``torch2phy`` defines the target normalized space
        (see :func:`_fireants_field_to_nib`).
    units : {'ras', 'vox'}
        Units of the components of ``data``.
    index : int
        Batch index to select ``torch2phy`` from.

    Returns
    -------
    torch.Tensor
        Field of shape ``(1, Z, Y, X, 3)`` in FireANTs' normalized ``[-1, 1]``
        torch coordinates, on ``image``'s device.
    """
    data = _from_units(data, affine, units)

    # (x, y, z, 3) -> (z, y, x, 3)
    data = np.transpose(data, (2, 1, 0, 3))

    # components: RAS -> LPS mm -> normalized
    data = data @ _RAS_TO_LPS[:3, :3].T
    t2p = _fireants_t2p(image, index=index)
    data = data @ np.linalg.inv(t2p[:3, :3]).T

    return torch.from_numpy(data[None]).to(device=image.device, dtype=torch.float32)


def _fireants_t2p(image, index: int = 0) -> np.ndarray:
    """Return the ``torch2phy`` matrix (normalized coords -> LPS mm) of ``image``.

    Parameters
    ----------
    image : fireants.io.image.Image or fireants.io.image.BatchedImages
        Both expose a ``torch2phy`` tensor of shape ``(N, 4, 4)`` (``N=1`` for
        a plain ``Image``).
    index : int
        Batch index to select.
    """
    t2p = image.torch2phy.detach().cpu().numpy().astype(np.float64)
    return t2p[min(index, t2p.shape[0] - 1)]


def _fireants_grid_affine(
        image, tensor_shape: Sequence[int], index: int = 0
) -> np.ndarray:
    """Build the RAS affine of a FireANTs grid of tensor shape ``(z, y, x)``.

    Normalized coordinates are resolution independent (``align_corners=True``),
    so the voxel-to-world affine of a grid of shape ``S`` is ``torch2phy``
    composed with ``index -> 2 * index / (S - 1) - 1``.
    """
    t2p = _fireants_t2p(image, index=index)

    size = np.asarray(tuple(tensor_shape)[::-1], dtype=np.float64)  # ITK (x, y, z)
    idx2torch = np.eye(4)
    idx2torch[:3, :3] = np.diag(2.0 / (size - 1))
    idx2torch[:3, 3] = -1.0

    return _RAS_TO_LPS @ (t2p @ idx2torch)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_units(data: np.ndarray, affine: np.ndarray, units: str) -> np.ndarray:
    """Convert components from RAS mm to ``units``."""
    if units == "ras":
        return data
    if units == "vox":
        return data @ np.linalg.inv(affine[:3, :3]).T
    raise ValueError(f"Unknown units '{units}', expected 'ras' or 'vox'.")


def _from_units(data: np.ndarray, affine: np.ndarray, units: str) -> np.ndarray:
    """Convert components from ``units`` to RAS mm."""
    if units == "ras":
        return data
    if units == "vox":
        return data @ affine[:3, :3].T
    raise ValueError(f"Unknown units '{units}', expected 'ras' or 'vox'.")