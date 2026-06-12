from typing import Optional, Union, Sequence, Any
import csv
import pdb
from os.path import join

import nibabel as nib
import numpy as np
from scipy.special import softmax
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.interpolate import RegularGridInterpolator as rgi
from munkres import Munkres


def align_with_identity_vox2ras0(V: np.ndarray, vox2ras0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Permute and flip a volume so its voxel-to-RAS matrix is close to identity.

    Uses the Hungarian algorithm to find the axis permutation that best
    aligns each voxel axis with the nearest RAS axis, then flips any axis
    whose diagonal entry in the resulting matrix is negative.

    Parameters
    ----------
    V : np.ndarray
        Input 3D volume, shape ``(X, Y, Z)``.
    vox2ras0 : np.ndarray
        Voxel-to-RAS affine matrix, shape ``(4, 4)``.

    Returns
    -------
    V : np.ndarray
        Re-oriented volume.
    v2r : np.ndarray
        Updated voxel-to-RAS affine matrix, shape ``(4, 4)``.
    """
    COST = np.zeros((3,3))
    for i in range(3):
        for j in range(3):

            # worker is the vector
            b = vox2ras0[:3,i]

            # task is j:th axis
            a = np.zeros((3,1))
            a[j] = 1

            COST[i, j] = - np.abs(np.dot(a.T, b))/np.linalg.norm(a, 2)/np.linalg.norm(b, 2)

    m = Munkres()
    indexes = m.compute(COST)

    v2r = np.zeros_like(vox2ras0)
    for idx in indexes:
        v2r[:, idx[1]] = vox2ras0[:, idx[0]]
    v2r[:, 3] = vox2ras0[:, 3]
    V = np.transpose(V, axes=[idx[1] for idx in indexes])

    for d in range(3):
        if v2r[d,d] < 0:
            v2r[:3, d] = -v2r[:3, d]
            v2r[:3, 3] = v2r[:3, 3] - v2r[:3, d] * (V.shape[d] -1)
            V = np.flip(V, axis=d)

    return V, v2r

def rescale_volume(volume: np.ndarray, new_min: float = 0, new_max: float = 255, min_percentile: float = 2,
                    max_percentile: float = 98, use_positive_only: bool = True) -> np.ndarray:
    """Linearly rescale a volume to a new intensity range.

    Parameters
    ----------
    volume : np.ndarray
        Input volume.
    new_min : float, optional
        Minimum value of the output range. Default is 0.
    new_max : float, optional
        Maximum value of the output range. Default is 255.
    min_percentile : float, optional
        Percentile used to estimate the robust minimum (0 = ``np.min``).
        Default is 2.
    max_percentile : float, optional
        Percentile used to estimate the robust maximum (100 = ``np.max``).
        Default is 98.
    use_positive_only : bool, optional
        If ``True``, percentiles are computed from positive voxels only.
        Default is ``True``.

    Returns
    -------
    np.ndarray
        Rescaled volume clipped to ``[robust_min, robust_max]`` and linearly
        mapped to ``[new_min, new_max]``.
    """

    # select only positive intensities
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

    # define min and max intensities in original image for normalisation
    robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) if max_percentile == 0 else np.percentile(intensities, max_percentile)

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * new_max
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)

def rescale_flow(flow_vol: np.ndarray, aff: np.ndarray, new_vox_size: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Rescale a displacement field to a new voxel size.

    Adjusts displacement vector magnitudes to account for the change in voxel
    spacing, then resamples the field to the target voxel size.

    Parameters
    ----------
    flow_vol : np.ndarray
        Displacement field, shape ``(X, Y, Z, 3)``, in voxel units.
    aff : np.ndarray
        Affine matrix of the input field, shape ``(4, 4)``.
    new_vox_size : array-like of float
        Target voxel size in mm, length 3.

    Returns
    -------
    flow_vol : np.ndarray
        Resampled displacement field in the new voxel space.
    flow_aff : np.ndarray
        Updated affine matrix, shape ``(4, 4)``.
    """
    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    f_factor = pixdim / new_vox_size

    flow_vol[..., 0] *= f_factor[0]
    flow_vol[..., 1] *= f_factor[1]
    flow_vol[..., 2] *= f_factor[2]

    flow_vol, flow_aff = rescale_voxel_size(flow_vol, aff, new_vox_size, not_aliasing=True)

    return flow_vol, flow_aff

def gaussian_smoothing_voxel_size(proxy: nib.Nifti1Image, new_vox_size: Sequence[float]) -> np.ndarray:
    """Apply Gaussian anti-aliasing smoothing before resampling to a new voxel size.

    Computes per-axis sigma values from the ratio of current to target
    voxel sizes. Axes being upsampled (factor > 1) receive zero smoothing.

    Parameters
    ----------
    proxy : nibabel.Nifti1Image
        Input image proxy providing the affine and data.
    new_vox_size : array-like of float
        Target voxel size in mm, length 3.

    Returns
    -------
    np.ndarray
        Smoothed image array.
    """
    pixdim = np.sqrt(np.sum(proxy.affine * proxy.affine, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume = np.array(proxy.dataobj)
    if len(volume.shape) > 3:
        sigmas = np.concatenate((sigmas, [0]))

    return gaussian_filter(volume, sigmas)

def rescale_voxel_size(volume: np.ndarray, aff: np.ndarray, new_vox_size: Sequence[float],
                        not_aliasing: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Resample a volume to a new voxel size using trilinear interpolation.

    Optionally applies Gaussian anti-aliasing before downsampling.
    The returned affine matrix is updated so that the RAS coordinates of
    the volume are preserved.

    Parameters
    ----------
    volume : np.ndarray
        Input volume array.
    aff : np.ndarray
        Affine (voxel-to-RAS) matrix of the input volume, shape ``(4, 4)``.
    new_vox_size : array-like of float
        Target voxel size in mm, length 3.
    not_aliasing : bool, optional
        If ``True``, skip the Gaussian anti-aliasing step even when
        downsampling. Default is ``False``.

    Returns
    -------
    volume2 : np.ndarray
        Resampled volume at the new voxel size.
    aff2 : np.ndarray
        Updated affine matrix, shape ``(4, 4)``.
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    if len(volume.shape) > 3:
        sigmas = np.concatenate((sigmas, [0]))

    if all(sigmas == 0) or not_aliasing:
        volume_filt = volume
    else:
        volume_filt = gaussian_filter(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = rgi((x, y, z), volume_filt)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape[:3] * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2

def gaussian_antialiasing(volume: np.ndarray, aff: np.ndarray, new_voxel_size: Sequence[float]) -> np.ndarray:
    """Apply a Gaussian anti-aliasing filter before downsampling a volume.

    Axes being upsampled (factor > 1) receive zero smoothing.

    Parameters
    ----------
    volume : np.ndarray
        Input volume array.
    aff : np.ndarray
        Affine matrix of the input volume, shape ``(4, 4)``.
    new_voxel_size : array-like of float
        Target voxel size in mm, length 3.

    Returns
    -------
    np.ndarray
        Gaussian-smoothed volume, same shape as ``volume``.
    """
    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_voxel_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    return gaussian_filter(volume, sigmas)

def get_rigid_params(matrix: Any, proxyref: nib.Nifti1Image,
                      cog: Optional[Sequence[float]] = None) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a rigid affine matrix into Euler angles and translation.

    Parameters
    ----------
    matrix : torch.Tensor
        4×4 rigid transformation matrix (rotation + translation block).
    proxyref : nibabel.Nifti1Image
        Reference image used to compute the center of rotation when ``cog``
        is ``None``.
    cog : array-like of float, optional
        Center of gravity in RAS mm. If ``None``, the image center is used.

    Returns
    -------
    angles : np.ndarray
        Euler angles ``[rx, ry, rz]`` in radians.
    translation : np.ndarray
        Translation vector ``[tx, ty, tz]`` in mm.
    """
    ry = -np.asin(matrix[2, 0])
    rx = np.atan2(matrix[2, 1] / np.cos(ry), matrix[2, 2] / np.cos(ry))
    rz = np.atan2(matrix[1, 0] / np.cos(ry), matrix[0, 0] / np.cos(ry))
    angles = np.array([rx, ry, rz])

    T_center = np.zeros((4, 4)).to(matrix.device)
    T_center[0, 0] = 1
    T_center[1, 1] = 1
    T_center[2, 2] = 1
    T_center[3, 3] = 1
    if cog is None:
        T_center[:3, 3] = (-proxyref.affine @ np.asarray([i / 2 for i in proxyref.shape] + [1]))[:3]
    else:
        T_center[:3, 3] = -cog


    T_center_inv = np.zeros((4, 4)).to(matrix.device)
    T_center_inv[0, 0] = 1
    T_center_inv[1, 1] = 1
    T_center_inv[2, 2] = 1
    T_center_inv[3, 3] = 1
    if cog is None:
        T_center_inv[:3, 3] = (proxyref.affine @ np.asarray([i / 2 for i in proxyref.shape] + [1]))[:3]
    else:
        T_center_inv[:3, 3] = cog

    T_rot = np.eye(4)
    T_rot[:3, :3] = matrix[:3, :3]
    T_trans = matrix @ T_center_inv @ np.linalg.inv(T_rot) @ T_center
    translation = T_trans[:3, 3]

    return angles, translation

def one_hot_encoding(target: np.ndarray, num_classes: Optional[int] = None,
                      categories: Optional[Union[dict, list, np.ndarray]] = None) -> np.ndarray:
    """Convert an integer label map to a one-hot encoded array.

    Parameters
    ----------
    target : np.ndarray
        Integer label map of shape ``(d1, d2, ..., dN)``.
    num_classes : int, optional
        Number of classes. Required when ``categories`` is ``None``.
    categories : dict, list, or np.ndarray, optional
        Defines the mapping from label value to channel index.

        - If a ``dict``, used directly as ``{label: channel_index}``.
        - If a list or array, converted to ``{label: i}`` by enumeration.
        - If ``None`` and ``num_classes`` is also ``None``, inferred from
          ``np.unique(target)``.

    Returns
    -------
    np.ndarray
        One-hot array of shape ``(d1, d2, ..., dN, num_classes)``,
        dtype ``uint16``.
    """

    if categories is None and num_classes is None:
        categories = {cls: it_cls for it_cls, cls in enumerate(np.sort(np.unique(target)))}
        num_classes = len(categories)

    elif categories is not None:
        if isinstance(categories, list) or isinstance(categories, np.ndarray):
            categories = {cls: it_cls for it_cls, cls in enumerate(categories)}

        num_classes = len(np.unique(list(categories.values())))

    else:
        categories = {cls: cls for cls in np.arange(num_classes)}

    labels = np.zeros((num_classes,) + target.shape, dtype='uint16')
    for cls, it_cls in categories.items():
        idx_class = np.where(target == cls)
        idx = (it_cls,) + idx_class
        labels[idx] = 1

    return np.transpose(labels, axes=(1, 2, 3, 0))

def label_log_odds(target: np.ndarray, num_classes: Optional[int] = None,
                    categories: Optional[Union[list, np.ndarray]] = None) -> np.ndarray:
    """Compute per-label signed distance maps as log-odds representations.

    For each label, the signed Euclidean distance transform is computed:
    positive inside the label region, negative outside. A large negative
    sentinel value (``-10000``) is used outside the bounding box.

    Parameters
    ----------
    target : np.ndarray
        Integer label map of shape ``(d1, d2, ..., dN)``.
    num_classes : int, optional
        Number of classes. Required when ``categories`` is ``None``.
    categories : list or np.ndarray, optional
        Ordered label values. If ``None`` and ``num_classes`` is also
        ``None``, inferred from ``np.unique(target)``.

    Returns
    -------
    np.ndarray
        Signed distance map of shape ``(num_classes, d1, d2, ..., dN)``,
        dtype ``int``.
    """

    if categories is None and num_classes is None:
        categories = np.sort(np.unique(target))
        num_classes = len(categories)

    elif categories is not None:
        num_classes = len(categories)

    else:
        categories = np.arange(num_classes)

    labels = -10000 * np.ones((num_classes,) + target.shape, dtype='int')
    for it_cls, cls in enumerate(categories):
        mask_label = target == cls
        bbox_label, crop_coord = crop_label(mask_label, margin=10)

        d_in = (distance_transform_edt(bbox_label))
        d_out = -distance_transform_edt(~bbox_label)
        d = np.zeros_like(d_in)
        d[bbox_label] = d_in[bbox_label]
        d[~bbox_label] = d_out[~bbox_label]

        labels[it_cls, crop_coord[0][0]: crop_coord[0][1], crop_coord[1][0]: crop_coord[1][1], crop_coord[2][0]: crop_coord[2][1]] = d

    return labels

def crop_label(mask: np.ndarray, margin: Union[int, list[int]] = 10,
                threshold: float = 0) -> tuple[np.ndarray, list[list[int]]]:
    """Crop a binary mask to its bounding box with an optional margin.

    Parameters
    ----------
    mask : np.ndarray
        3D array; voxels above ``threshold`` are treated as foreground.
    margin : int or list of int, optional
        Number of voxels to expand the bounding box in each dimension.
        A single int applies the same margin to all dimensions. Default is 10.
    threshold : float, optional
        Voxels with values strictly above this are considered foreground.
        Default is 0.

    Returns
    -------
    mask_cropped : np.ndarray
        Cropped sub-volume of the mask.
    crop_coord : list of [int, int]
        Crop coordinates ``[[x0, x1], [y0, y1], [z0, z1]]``.
    """
    ndim = len(mask.shape)
    if isinstance(margin, int):
        margin=[margin]*ndim

    crop_coord = []
    idx = np.where(mask>threshold)
    for it_index, index in enumerate(idx):
        clow = max(0, np.min(idx[it_index]) - margin[it_index])
        chigh = min(mask.shape[it_index], np.max(idx[it_index]) + margin[it_index])
        crop_coord.append([clow, chigh])

    mask_cropped = mask[
                   crop_coord[0][0]: crop_coord[0][1],
                   crop_coord[1][0]: crop_coord[1][1],
                   crop_coord[2][0]: crop_coord[2][1]
                   ]

    return mask_cropped, crop_coord

def apply_crop(image: np.ndarray, crop_coord: list[list[int]]) -> np.ndarray:
    """Extract a sub-volume defined by crop coordinates.

    Parameters
    ----------
    image : np.ndarray
        3D input array.
    crop_coord : list of [int, int]
        Crop coordinates as returned by :func:`crop_label`.

    Returns
    -------
    np.ndarray
        Cropped sub-volume.
    """
    return image[crop_coord[0][0]: crop_coord[0][1],
                 crop_coord[1][0]: crop_coord[1][1],
                 crop_coord[2][0]: crop_coord[2][1]
           ]

def compute_centroids_ras(seg_file: str, labelfile: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute RAS-space centroids for each label in a segmentation.

    Labels with fewer than 50 voxels are flagged as missing.

    Parameters
    ----------
    seg_file : str
        Path to a NIfTI segmentation file.
    labelfile : str
        Path to a ``.npy`` file containing the integer label values of interest.

    Returns
    -------
    refCOG : np.ndarray
        Image centroids in RAS coordinates (mm) , shape ``(3, n_labels)``
    ok : np.ndarray
        Binary flag array of length ``n_labels``; 1 if the label had ≥ 50
        voxels, 0 otherwise.
    """
    seg_proxy = nib.load(seg_file)
    seg_buffer = np.array(seg_proxy.dataobj)
    labels = np.load(labelfile)

    nlab = len(labels)
    ref_cog = np.zeros([4, nlab])

    ok = np.ones(nlab)
    for l in range(nlab):
        aux = np.where(seg_buffer == labels[l])
        if len(aux[0]) > 50:
            ref_cog[0, l] = np.median(aux[0])
            ref_cog[1, l] = np.median(aux[1])
            ref_cog[2, l] = np.median(aux[2])
            ref_cog[3, l] = 1
        else:
            ok[l] = 0

    ref_cog = np.matmul(seg_proxy.affine, ref_cog)[:-1, :]

    return ref_cog, ok

def compute_distance_map_nongrid(labelmap: np.ndarray, sampling_grid: np.ndarray,
                                  labels_lut: Optional[dict] = None) -> np.ndarray:
    """Compute signed distance maps evaluated on an arbitrary sampling grid.

    For each label, the signed Euclidean distance is computed: positive
    inside the label and negative outside.

    Parameters
    ----------
    labelmap : np.ndarray
        Integer label map, shape ``(X, Y, Z)``.
    sampling_grid : np.ndarray
        Sampling coordinates, shape ``(3, N1, N2, ...)``, in voxel units.
    labels_lut : dict, optional
        Mapping ``{label_value: channel_index}``. If ``None``, inferred
        from ``np.unique(labelmap)``.

    Returns
    -------
    np.ndarray
        Signed distance map, shape ``(*sampling_grid.shape[1:], n_labels)``,
        dtype ``float32``.
    """
    if labels_lut is None:
        labels_lut = {ul: it_ul for it_ul, ul in enumerate(np.unique(labelmap))}

    sampling_grid_nn = np.round(sampling_grid).astype('int')
    sampling_grid_nn[0] = np.clip(sampling_grid_nn[0], 0, labelmap.shape[0] - 1)
    sampling_grid_nn[1] = np.clip(sampling_grid_nn[1], 0, labelmap.shape[1] - 1)
    sampling_grid_nn[2] = np.clip(sampling_grid_nn[2], 0, labelmap.shape[2] - 1)
    distancemap = -200 * np.ones(sampling_grid_nn.shape[1:] + (len(labels_lut.keys()),), dtype='float32')
    for ul, it_ul in labels_lut.items():

        mask_label = labelmap == ul
        mask_label_reg = mask_label[sampling_grid_nn[0], sampling_grid_nn[1], sampling_grid_nn[2]]
        if np.sum(mask_label) == 0:
            continue
        else:

            idx_in = distance_transform_edt(mask_label, return_distances=False, return_indices=True)
            d_in = np.sqrt(np.sum((idx_in[:, sampling_grid_nn[0], sampling_grid_nn[1], sampling_grid_nn[2]] - sampling_grid)**2, axis=0))
            idx_out = distance_transform_edt(~mask_label,  return_distances=False, return_indices=True)
            d_out = -np.sqrt(np.sum((idx_out[:, sampling_grid_nn[0], sampling_grid_nn[1], sampling_grid_nn[2]] - sampling_grid)**2, axis=0))

            # With crop (it is approximate near the boundaries)
            # bbox_label, crop_coord = crop_label(mask_label, margin=5)
            # idx_in = distance_transform_edt(bbox_label, return_distances=False, return_indices=True)
            # d_in = np.sqrt(np.sum((idx_in[:,
            #                        np.clip(sampling_grid_nn[0]-crop_coord[0][0], 0, bbox_label.shape[0]-1),
            #                        np.clip(sampling_grid_nn[1]-crop_coord[1][0], 0, bbox_label.shape[1]-1),
            #                        np.clip(sampling_grid_nn[2]-crop_coord[2][0], 0, bbox_label.shape[2]-1)]
            #                        - sampling_grid + np.array([crop_coord[0][0], crop_coord[1][0], crop_coord[2][0]]).reshape((3, 1, 1, 1))) ** 2, axis=0))
            # idx_out = distance_transform_edt(~bbox_label,  return_distances=False, return_indices=True)
            # d_out = - np.sqrt(np.sum((idx_out[:,
            #                           np.clip(sampling_grid_nn[0]-crop_coord[0][0], 0, bbox_label.shape[0]-1),
            #                           np.clip(sampling_grid_nn[1]-crop_coord[1][0], 0, bbox_label.shape[1]-1),
            #                           np.clip(sampling_grid_nn[2]-crop_coord[2][0], 0, bbox_label.shape[2]-1)]
            #                           - sampling_grid + np.array([crop_coord[0][0], crop_coord[1][0], crop_coord[2][0]]).reshape((3, 1, 1, 1))) ** 2, axis=0))

            d = np.zeros_like(d_in)
            d[mask_label_reg] = d_in[mask_label_reg]
            d[~mask_label_reg] = d_out[~mask_label_reg]
            distancemap[... , it_ul] = d

    return distancemap

def compute_distance_map(labelmap: np.ndarray, soft_seg: bool = True, labels_lut: Optional[dict] = None) -> np.ndarray:
    """Compute signed distance maps on the native label grid.

    For each label the computation is cropped to a tight bounding box with
    a 5-voxel margin to reduce memory usage.

    Parameters
    ----------
    labelmap : np.ndarray
        Integer label map, shape ``(X, Y, Z)``.
    soft_seg : bool, optional
        If ``True``, apply softmax along the label axis before returning.
        Default is ``True``.
    labels_lut : dict, optional
        Mapping ``{label_value: channel_index}``. If ``None``, inferred
        from ``np.unique(labelmap)``.

    Returns
    -------
    np.ndarray
        Distance map, shape ``(X, Y, Z, n_labels)``, dtype ``float32``.
    """
    if labels_lut is None:
        labels_lut = {ul: it_ul for it_ul, ul in enumerate(np.unique(labelmap))}

    distancemap = -200 * np.ones(labelmap.shape + (len(labels_lut.keys()),), dtype='float32')
    for ul, it_ul in labels_lut.items():

        mask_label = labelmap == ul
        if np.sum(mask_label) == 0:
            continue
        else:
            mask_label, crop_coord = crop_label(mask_label, margin=5)

            d_in = (distance_transform_edt(mask_label))
            d_out = -distance_transform_edt(~mask_label)
            d = np.zeros_like(d_in)
            d[mask_label] = d_in[mask_label]
            d[~mask_label] = d_out[~mask_label]

            distancemap[
                crop_coord[0][0]: crop_coord[0][1],
                crop_coord[1][0]: crop_coord[1][1],
                crop_coord[2][0]: crop_coord[2][1], it_ul
            ] = d

    # distancemap = np.clip(distancemap, -3, np.max(distancemap))

    if soft_seg:
        distancemap = softmax(distancemap, axis=-1)

    return distancemap

def compute_distance_map_crop(labelmap: np.ndarray, soft_seg: bool = True,
                               labels_lut: Optional[dict] = None) -> np.ndarray:
    """Compute signed distance maps using per-label bounding-box crops.

    Equivalent to :func:`compute_distance_map` but operates on cropped
    sub-volumes named ``bbox_label`` rather than the full label mask,
    which can be faster for sparse labels.

    Parameters
    ----------
    labelmap : np.ndarray
        Integer label map, shape ``(X, Y, Z)``.
    soft_seg : bool, optional
        If ``True``, apply softmax along the label axis. Default is ``True``.
    labels_lut : dict, optional
        Mapping ``{label_value: channel_index}``. If ``None``, inferred
        from ``np.unique(labelmap)``.

    Returns
    -------
    np.ndarray
        Distance map, shape ``(X, Y, Z, n_labels)``, dtype ``float32``.
    """
    if labels_lut is None:
        labels_lut = {ul: it_ul for it_ul, ul in enumerate(np.unique(labelmap))}

    distancemap = -200 * np.ones(labelmap.shape + (len(labels_lut.keys()),), dtype='float32')
    for ul, it_ul in labels_lut.items():

        mask_label = labelmap == ul
        if np.sum(mask_label) == 0:
            continue
        else:
            bbox_label, crop_coord = crop_label(mask_label, margin=5)

            d_in = (distance_transform_edt(bbox_label))
            d_out = -distance_transform_edt(~bbox_label)
            d = np.zeros_like(d_in)
            d[bbox_label] = d_in[bbox_label]
            d[~bbox_label] = d_out[~bbox_label]

            distancemap[crop_coord[0][0]: crop_coord[0][1],
                        crop_coord[1][0]: crop_coord[1][1],
                        crop_coord[2][0]: crop_coord[2][1], it_ul] = d

    if soft_seg:
        distancemap = softmax(distancemap, axis=-1)

    return distancemap


