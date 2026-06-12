from typing import TypedDict, Optional, Union, Any, Sequence
import pdb
from datetime import datetime, date
from os.path import join, exists, basename, dirname
from os import makedirs, listdir
import os
import gc
import subprocess
import json
import shutil

import nibabel as nib
from skimage.transform import rescale
import numpy as np
import csv


class ProcessResult(TypedDict, total=False):
    exit_code: int
    message: str
    data: Optional[dict]  # optional payload


def get_bids_entities(fname: str) -> dict:
    """Parse BIDS entities from a filename into a dictionary.

    Parameters
    ----------
    fname : str
        BIDS-formatted filename (e.g.
        ``sub-001_ses-01_T1w.nii.gz``).

    Returns
    -------
    dict
        Mapping of entity keys to values. The ``'subject'`` and
        ``'session'`` keys are normalised from the BIDS short forms
        ``'sub'`` and ``'ses'``. A ``'suffix'`` and ``'extension'``
        key are also set for the final filename component.
    """
    edict = {}
    fparts = basename(fname).split("_")
    for f in fparts:

        if "." in f:
            edict["suffix"] = f.split(".")[0]
            edict["extension"] = ".".join(f.split(".")[1:])

        else:
            e, v = f.split("-")
            if e == "sub":
                e = "subject"
            elif e == "ses":
                e = "session"

            edict["suffix"] = e
            edict[e] = v

    return edict


def save_nii(im: np.ndarray, v2r: np.ndarray, filepath: str) -> None:
    """Save a NumPy array as a NIfTI file, creating parent directories as needed.

    Boolean arrays are cast to ``uint8`` and ``int64`` arrays to ``float``
    before saving to ensure NIfTI compatibility.

    Parameters
    ----------
    im : np.ndarray
        Image data to save.
    v2r : np.ndarray
        Voxel-to-RAS affine matrix, shape ``(4, 4)``.
    filepath : str
        Output file path (e.g. ``path/to/image.nii.gz``).
    """
    if im.dtype in [np.bool_]:
        im = im.astype("uint8")
    elif im.dtype in [np.int64]:
        im = im.astype("float")

    create_dir(dirname(filepath))
    img = nib.Nifti1Image(im, v2r)
    nib.save(img, filepath)


def check_nan(v: Union[str, float]) -> bool:
    """Return whether a value represents a missing / NaN observation.

    Parameters
    ----------
    v : str or float
        Value to test.

    Returns
    -------
    bool
        ``True`` if ``v`` is ``''``, ``'nan'``, ``'NaN'``, or
        ``float('nan')``.
    """
    if v in ["", "nan", "NaN"] or np.isnan(v):
        return True
    else:
        return False


def create_dir(path: str, subdirs: Optional[Union[str, list[str]]] = None) -> None:
    """Create a directory (and optional sub-directories) if they do not exist.

    Parameters
    ----------
    path : str
        Root directory to create.
    subdirs : str or list of str, optional
        Sub-directory name(s) to create inside ``path``. If ``None``,
        only ``path`` itself is created.
    """
    if subdirs is None:
        if not exists(path):
            makedirs(path)
    else:
        if isinstance(subdirs, str):
            subdirs = [subdirs]

        for subdir in subdirs:
            if not exists(join(path, subdir)):
                makedirs(join(path, subdir))


def remove_dir(path: str) -> None:
    """Recursively remove a directory and all its contents.

    Parameters
    ----------
    path : str
        Path to the directory to remove.
    """
    subprocess.call(["rm", "-rf", path])


def create_derivative_dir(path: str, description: str) -> None:
    """Create a BIDS derivative directory with a ``dataset_description.json``.

    The JSON file is only written if it does not already exist.

    Parameters
    ----------
    path : str
        Path to the derivative directory to create.
    description : str
        Human-readable description written to the ``'Description'`` field
        of ``dataset_description.json``.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    data_descr_path = os.path.join(path, "dataset_description.json")

    if not os.path.exists(data_descr_path):
        data_descr = {}
        data_descr["Name"] = os.path.basename(path)
        data_descr["BIDSVersion"] = "1.0.2"
        data_descr["GeneratedBy"] = [{"Name": os.path.basename(path)}]
        data_descr["Description"] = description
        json_object = json.dumps(data_descr, indent=4)
        with open(data_descr_path, "w") as outfile:
            outfile.write(json_object)


def write_json_derivatives(
    pixdim: Sequence[float],
    volshape: Sequence[int],
    filename: str,
    extra_kwargs: dict = {},
) -> None:
    """Write a BIDS-style JSON sidecar with image resolution and shape.

    Parameters
    ----------
    pixdim : array-like of float
        Voxel sizes along R, A, S axes in mm (length ≥ 3).
    volshape : array-like of int
        Image dimensions along X, Y, Z axes (length ≥ 3).
    filename : str
        Output ``.json`` file path.
    extra_kwargs : dict, optional
        Additional key-value pairs to merge into the JSON object.
    """
    im_json = {
        "Resolution": {"R": str(pixdim[0]), "A": str(pixdim[1]), "S": str(pixdim[2])},
        "ImageShape": {
            "X": str(volshape[0]),
            "Y": str(volshape[1]),
            "Z": str(volshape[2]),
        },
    }

    im_json = {**im_json, **extra_kwargs}

    json_object = json.dumps(im_json, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(json_object)


def create_results_dir(results_dir: str, subdirs: Optional[list[str]] = None) -> None:
    """Create a results directory with standard sub-directories.

    Parameters
    ----------
    results_dir : str
        Root results directory to create.
    subdirs : list of str, optional
        Sub-directory names to create. Defaults to
        ``['checkpoints', 'results']``.
    """
    if subdirs is None:
        subdirs = ["checkpoints", "results"]
    if not exists(results_dir):
        for sd in subdirs:
            makedirs(join(results_dir, sd))
    else:
        for sd in subdirs:
            if not exists(join(results_dir, sd)):
                makedirs(join(results_dir, sd))


def mri_convert_nifti_directory(directory: str, extension: str = ".nii.gz") -> None:
    """Convert all NIfTI / MGZ files in a directory to a target format via ``mri_convert``.

    Files already in the target ``extension`` are skipped.

    Parameters
    ----------
    directory : str
        Path to the directory containing the images to convert.
    extension : str, optional
        Target file extension. Default is ``'.nii.gz'``.
    """
    files = listdir(directory)
    for f in files:
        if extension in f:
            continue
        elif ".nii.gz" in f:
            new_f = f[:-6] + extension

        elif ".nii" in f:
            new_f = f[:-4] + extension

        elif ".mgz" in f:
            new_f = f[:-4] + extension

        else:
            continue

        subprocess.call(["mri_convert", join(directory, f), join(directory, new_f)])


def read_lta(file: str) -> np.ndarray:
    """Read a FreeSurfer LTA (linear transform array) file into a 4×4 matrix.

    Parameters
    ----------
    file : str
        Path to the ``.lta`` file.

    Returns
    -------
    np.ndarray
        Affine transformation matrix, shape ``(4, 4)``.
    """
    lta = np.zeros((4, 4))
    with open(file, "r") as txtfile:
        lines = txtfile.readlines()
        for it_row, l in enumerate(lines[5:9]):
            aff_row = l.split(" ")[:-1]
            lta[it_row] = [float(ar) for ar in aff_row]

    return lta


def read_tsv(
    path: str,
    k: Optional[Union[str, list[str]]] = None,
    delimiter: str = "\t",
    codec_read: bool = False,
) -> dict:
    """Read a delimited file into a dictionary keyed by one or more columns.

    Parameters
    ----------
    path : str
        Path to the TSV (or CSV) file.
    k : str or list of str, optional
        Column name(s) to use as the dictionary key. Multiple columns are
        joined with ``'_'``. Defaults to the first column in the file.
    delimiter : str, optional
        Field delimiter. Default is ``'\\t'``.
    codec_read : bool, optional
        Reserved for future use. Currently unused. Default is ``False``.

    Returns
    -------
    dict
        Mapping from key string to the full row dictionary.
    """
    t = {}
    with open(path, "r") as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=delimiter)
        if k is None:
            k = csvreader.fieldnames[0]
        if not isinstance(k, list):
            k = [k]

        for it_row, row in enumerate(csvreader):
            t["_".join([row[i] for i in k])] = row

    return t


def write_affine_matrix(path: str, affine_matrix: np.ndarray) -> None:
    """Write a 4×4 affine matrix to a space-delimited text file.

    Parameters
    ----------
    path : str
        Output file path.
    affine_matrix : np.ndarray
        Affine matrix, shape ``(4, 4)``.
    """
    with open(path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=" ")
        for it_row in range(4):
            csvwriter.writerow(affine_matrix[it_row])


def read_affine_matrix(
    path: str, full: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Read an affine matrix from a space-delimited text file.

    Parameters
    ----------
    path : str
        Path to the file produced by :func:`write_affine_matrix`.
    full : bool, optional
        If ``True``, return the full 4×4 affine matrix. If ``False``
        (default), return the 3×3 rotation block and the 3-element
        translation vector separately.

    Returns
    -------
    affine_matrix : np.ndarray
        Shape ``(4, 4)`` when ``full=True``.
    rotation_matrix : np.ndarray
        Shape ``(3, 3)`` when ``full=False``.
    translation_vector : np.ndarray
        Shape ``(3,)`` when ``full=False``.
    """
    with open(path, "r") as csvfile:
        rotation_matrix = np.zeros((3, 3))
        translation_vector = np.zeros((3,))
        csvreader = csv.reader(csvfile, delimiter=" ")
        for it_row, row in enumerate(csvreader):
            rotation_matrix[it_row, 0] = float(row[0])
            rotation_matrix[it_row, 1] = float(row[1])
            rotation_matrix[it_row, 2] = float(row[2])
            translation_vector[it_row] = float(row[3])
            if it_row == 2:
                break

    if full:
        affine_matrix = np.zeros((4, 4))
        affine_matrix[:3, :3] = rotation_matrix
        affine_matrix[:3, 3] = translation_vector
        affine_matrix[3, 3] = 1
        return affine_matrix

    else:
        return rotation_matrix, translation_vector


def load_array_if_path(var: Any, load_as_numpy: bool = True) -> Any:
    """Load a NumPy array from a file path, or pass through the value unchanged.

    Parameters
    ----------
    var : str or any
        If a string and ``load_as_numpy`` is ``True``, treated as a file path
        and loaded with ``np.load``. Otherwise returned as-is.
    load_as_numpy : bool, optional
        Whether to attempt loading ``var`` as a NumPy file. Default is ``True``.

    Returns
    -------
    np.ndarray or any
        Loaded array, or the original ``var`` if no loading was performed.
    """
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), "No such path: %s" % var
        var = np.load(var)
    return var


def get_dims(shape: Sequence[int], max_channels: int = 10) -> tuple[int, int]:
    """Infer the number of spatial dimensions and channels from an array shape.

    If the last dimension is ≤ ``max_channels`` it is interpreted as a channel
    axis; otherwise the array is treated as purely spatial with one channel.

    Parameters
    ----------
    shape : sequence of int
        Shape of the array (e.g. ``volume.shape``).
    max_channels : int, optional
        Maximum value of the last dimension that is still considered a channel
        axis. Default is 10.

    Returns
    -------
    n_dims : int
        Number of spatial dimensions.
    n_channels : int
        Number of channels.

    Examples
    --------
    >>> get_dims([150, 150, 150], max_channels=10)
    (3, 1)
    >>> get_dims([150, 150, 150, 3], max_channels=10)
    (3, 3)
    >>> get_dims([150, 150, 150, 15], max_channels=10)
    (4, 1)
    """
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def get_ras_axes(aff: np.ndarray, n_dims: int = 3) -> np.ndarray:
    """Find which voxel axis corresponds to each RAS axis from an affine matrix.

    Parameters
    ----------
    aff : np.ndarray
        Affine matrix. Accepted shapes: ``(n_dims, n_dims)``,
        ``(n_dims+1, n_dims+1)``, or ``(n_dims, n_dims+1)``.
    n_dims : int, optional
        Number of spatial dimensions. Default is 3.

    Returns
    -------
    np.ndarray
        1D array of length ``n_dims`` mapping each RAS axis (R=0, A=1, S=2)
        to the corresponding voxel axis index.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i

    return img_ras_axes


def align_volume_to_ref(
    volume: np.ndarray,
    aff: np.ndarray,
    aff_ref: Optional[np.ndarray] = None,
    return_aff: bool = False,
    n_dims: Optional[int] = None,
    return_copy: bool = True,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Align a volume to the axis orientation and flip direction of a reference affine.

    Swaps and flips axes of ``volume`` so that its orientation matches
    ``aff_ref``, updating the affine accordingly.

    Parameters
    ----------
    volume : np.ndarray
        Floating volume to reorient.
    aff : np.ndarray
        Affine matrix of the floating volume, shape ``(4, 4)``.
    aff_ref : np.ndarray, optional
        Target orientation affine. Defaults to ``np.eye(4)`` (identity /
        RAS-aligned).
    return_aff : bool, optional
        If ``True``, also return the updated affine matrix. Default is
        ``False``.
    n_dims : int, optional
        Number of spatial dimensions. Inferred from ``volume.shape`` when
        ``None``.
    return_copy : bool, optional
        If ``True``, operate on a copy of the volume. Default is ``True``.

    Returns
    -------
    new_volume : np.ndarray
        Reoriented volume.
    aff_flo : np.ndarray
        Updated affine, shape ``(4, 4)``. Only returned when
        ``return_aff=True``.
    """

    # work on copy
    new_volume = volume.copy() if return_copy else volume
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = (
                ras_axes_flo[i],
                ras_axes_flo[swapped_axis_idx],
            )

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = -aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume


def reformat_to_list(
    var: Union[str, int, float, bool, list, tuple, np.ndarray, None],
    length: Optional[int] = None,
    load_as_numpy: bool = False,
    dtype: Optional[str] = None,
) -> Optional[list]:
    """Reformat a scalar, tuple, array, or string into a typed list of given length.

    Parameters
    ----------
    var : str, int, float, bool, list, tuple, or np.ndarray
        Value to reformat. ``None`` is returned unchanged.
    length : int, optional
        Desired list length. If ``var`` has a single element it is replicated;
        otherwise ``var`` must already have this length.
    load_as_numpy : bool, optional
        If ``True`` and ``var`` is a string, load it as a NumPy array first.
        Default is ``False``.
    dtype : {'int', 'float', 'bool', 'str'}, optional
        Convert every list element to this type after reformatting.

    Returns
    -------
    list or None
        Reformatted list, or ``None`` if ``var`` is ``None``.

    Raises
    ------
    ValueError
        If ``var`` is a list/array of length ≠ 1 and ≠ ``length``.
    TypeError
        If ``var`` cannot be converted to a list.
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(
        var, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)
    ):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError(
                    "if var is a list/tuple/numpy array, it should be of length 1 or {0}, "
                    "had {1}".format(length, var)
                )
    else:
        raise TypeError(
            "var should be an int, float, tuple, list, numpy array, or path to numpy array"
        )

    # convert items type
    if dtype is not None:
        if dtype == "int":
            var = [int(v) for v in var]
        elif dtype == "float":
            var = [float(v) for v in var]
        elif dtype == "bool":
            var = [bool(v) for v in var]
        elif dtype == "str":
            var = [str(v) for v in var]
        else:
            raise ValueError(
                "dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype)
            )
    return var


def load_volume(
    path_volume: str,
    im_only: bool = True,
    squeeze: bool = True,
    dtype: Optional[str] = None,
    aff_ref: Optional[np.ndarray] = None,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, Any]]:
    """Load a medical image volume from disk.

    Supported formats: ``.nii``, ``.nii.gz``, ``.mgz``, ``.npz``.
    For ``.npz`` files the array must be stored under the key
    ``'vol_data'``; the affine defaults to identity and the header is blank.

    Parameters
    ----------
    path_volume : str
        Path to the image file.
    im_only : bool, optional
        If ``True`` (default), return only the volume array. If ``False``,
        also return the affine matrix and header.
    squeeze : bool, optional
        Whether to squeeze singleton dimensions. Default is ``True``.
    dtype : str, optional
        NumPy dtype to cast the volume to after loading (e.g. ``'float32'``).
        Integer dtypes trigger rounding before casting.
    aff_ref : np.ndarray, optional
        If provided, align the volume to this 4×4 affine matrix before
        returning.

    Returns
    -------
    volume : np.ndarray
        Image data array.
    aff : np.ndarray
        Affine matrix, shape ``(4, 4)``. Only returned when
        ``im_only=False``.
    header : nibabel header
        Image header. Only returned when ``im_only=False``.
    """
    assert path_volume.endswith((".nii", ".nii.gz", ".mgz", ".npz")), (
        "Unknown data file: %s" % path_volume
    )

    if path_volume.endswith((".nii", ".nii.gz", ".mgz")):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)["vol_data"]
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if "int" in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
        volume, aff = align_volume_to_ref(
            volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims
        )

    if im_only:
        return volume
    else:
        return volume, aff, header


def save_volume(
    volume: np.ndarray,
    aff: np.ndarray | str | None,
    path: str,
    header: Optional[nib.Nifti1Header] = None,
    res: Optional[float | int] = None,
    dtype: Optional[str] = None,
    n_dims: int = 3,
):
    """Save a volume array to disk as a NIfTI or NPZ file.

    Parameters
    ----------
    volume : np.ndarray
        Image data to save.
    aff : np.ndarray or str or None
        Voxel-to-RAS affine matrix, shape ``(4, 4)``. Pass ``'FS'`` to use
        the standard FreeSurfer output affine, or ``None`` for identity.
    header : nibabel header or None
        Image header. A blank ``Nifti1Header`` is used when ``None``.
    path : str
        Output file path (``.nii``, ``.nii.gz``, or ``.npz``).
    res : float or array-like of float, optional
        If provided, update the voxel sizes in the header before saving.
    dtype : str, optional
        NumPy dtype for the saved data. Integer dtypes trigger rounding.
    n_dims : int, optional
        Number of spatial dimensions, used to interpret ``res``. Default is 3.
    """

    if not exists(os.path.dirname(path)):
        makedirs(os.path.dirname(path))
    if ".npz" in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == "FS":
                aff = np.array(
                    [[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
                )
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)
        if dtype is not None:
            if "int" in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
            nifty.set_data_dtype(dtype)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)


def get_volume_info(
    path_volume: str,
    return_volume: bool = False,
    aff_ref: Optional[np.ndarray] = None,
    max_channels: int = 10,
) -> tuple:
    """Gather metadata about a volume without necessarily returning the data.

    Parameters
    ----------
    path_volume : str
        Path to the image file.
    return_volume : bool, optional
        If ``True``, include the loaded array in the return tuple. Default is
        ``False``.
    aff_ref : np.ndarray, optional
        If provided, align the volume and recompute shape / resolution in this
        4×4 reference space before returning.
    max_channels : int, optional
        Passed to :func:`get_dims` to determine the channel threshold.
        Default is 10.

    Returns
    -------
    volume : np.ndarray
        Image data. Only included when ``return_volume=True``.
    im_shape : list of int
        Spatial shape (without channel dimension).
    aff : np.ndarray
        Original affine matrix, shape ``(4, 4)``.
    n_dims : int
        Number of spatial dimensions.
    n_channels : int
        Number of image channels.
    header : nibabel header
        Image header.
    data_res : np.ndarray
        Voxel sizes in mm, length ``n_dims``.
    """
    # read image
    im, aff, header = load_volume(path_volume, im_only=False)

    # understand if image is multichannel
    im_shape = list(im.shape)
    n_dims, n_channels = get_dims(im_shape, max_channels=max_channels)
    im_shape = im_shape[:n_dims]

    # get labels res
    if ".nii" in path_volume:
        data_res = np.array(header["pixdim"][1: n_dims + 1])
    elif ".mgz" in path_volume:
        data_res = np.array(header["delta"])  # mgz image
    else:
        data_res = np.array([1.0] * n_dims)

    # align to given affine matrix
    if aff_ref is not None:
        ras_axes = get_ras_axes(aff, n_dims=n_dims)
        ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
        im = align_volume_to_ref(im, aff, aff_ref=aff_ref, n_dims=n_dims)
        im_shape = np.array(im_shape)
        data_res = np.array(data_res)
        im_shape[ras_axes_ref] = im_shape[ras_axes]
        data_res[ras_axes_ref] = data_res[ras_axes]
        im_shape = im_shape.tolist()

    # return info
    if return_volume:
        return im, im_shape, aff, n_dims, n_channels, header, data_res
    else:
        return im_shape, aff, n_dims, n_channels, header, data_res


def get_run(file: str) -> str:
    """Extract the BIDS ``run`` index from a filename.

    Parameters
    ----------
    file : str
        BIDS filename (e.g. ``sub-001_ses-01_run-02_T1w.nii.gz``).

    Returns
    -------
    str
        The run index string (e.g. ``'02'``), or ``''`` if no run entity is
        present.
    """
    for f in file.split("_"):
        if "run" in f:
            return f.split("-")[1]
    return ""


def load_chunk(
    ps: nib.Nifti1Image, chunk: list[list[int]], ps_id: Any
) -> tuple[np.ndarray, Any]:
    """Load a spatial sub-volume (chunk) from a NIfTI proxy and add a batch axis.

    Parameters
    ----------
    ps : nibabel.Nifti1Image
        Source image proxy.
    chunk : list of [int, int]
        Crop coordinates ``[[x0, x1], [y0, y1], [z0, z1]]``.
    ps_id : any
        Identifier passed through as the second return value (e.g. subject ID).

    Returns
    -------
    data : np.ndarray
        Sub-volume with a leading batch dimension, shape ``(1, dx, dy, dz)``.
    ps_id : any
        The ``ps_id`` argument, unchanged.
    """
    return (
        np.asarray(
            ps.dataobj[
                chunk[0][0]: chunk[0][1],
                chunk[1][0]: chunk[1][1],
                chunk[2][0]: chunk[2][1],
            ]
        )[np.newaxis],
        ps_id,
    )


def load_chunk_rescale(
    ps: nib.Nifti1Image, chunk: list[list[int]], ps_id: Any, factor: float
) -> tuple[np.ndarray, Any]:
    """Load and spatially rescale a NIfTI volume, then return a chunk.

    Parameters
    ----------
    ps : nibabel.Nifti1Image
        Source image proxy.
    chunk : list of [int, int]
        Crop coordinates ``[[x0, x1], [y0, y1], [z0, z1]]`` applied after
        rescaling.
    ps_id : any
        Identifier passed through as the second return value.
    factor : float
        Isotropic rescaling factor applied to the first three dimensions.

    Returns
    -------
    data : np.ndarray
        Rescaled and cropped sub-volume with a leading batch dimension,
        shape ``(1, dx, dy, dz)``.
    ps_id : any
        The ``ps_id`` argument, unchanged.
    """
    svf = np.asarray(ps.dataobj)
    svf = rescale(svf, [factor] * 3 + [1])
    return (
        svf[
            chunk[0][0]: chunk[0][1],
            chunk[1][0]: chunk[1][1],
            chunk[2][0]: chunk[2][1],
        ][np.newaxis],
        ps_id,
    )


def get_age(age: Union[str, float]) -> Optional[float]:
    """Parse an age value to float, returning ``None`` for missing values.

    Parameters
    ----------
    age : str or float
        Age value, possibly ``'nan'`` or a non-numeric string.

    Returns
    -------
    float or None
        Parsed age, or ``None`` if parsing fails or the value is ``'nan'``.
    """
    if age == "nan":
        age = None
    try:
        age = float(age)
    except Exception:
        age = None

    return age


def get_dx_bl(subject: Any) -> int:
    """Extract the baseline diagnosis code from a subject object.

    Looks up ``'dx'`` then ``'diagnosis'`` in the subject dictionary, and
    falls back to deriving the longitudinal diagnosis from session metadata.

    Parameters
    ----------
    subject : object
        Subject object with attributes ``sbj_dict`` (dict) and ``sessions``
        (list of session objects with ``sess_metadata`` dicts).

    Returns
    -------
    int
        Diagnosis code as defined in :data:`DX_DICT`.
    """
    if "dx" in subject.sbj_dict.keys():
        dx = subject.sbj_dict["dx"]
    elif "diagnosis" in subject.sbj_dict.keys():
        dx = subject.sbj_dict["diagnosis"]
    else:
        dx = DX_DICT[
            get_dx_long(
                [
                    tp.sess_metadata["dx"] if "dx" in tp.sess_metadata.keys() else ""
                    for tp in subject.sessions
                ]
            )
        ]

    return dx


def get_dx(dx: str) -> int:
    """Map a diagnosis string to an integer code.

    Parameters
    ----------
    dx : str
        Diagnosis label. Recognised values include ``''``, ``'-1'``,
        ``'CN'``, ``'Control'``, ``'CTR'``, ``'MCI'``, ``'Dementia'``,
        ``'AD'``, ``'FTD'``, ``'DFT'``, and ``'PortadorGENFI'``.

    Returns
    -------
    int
        Integer code: ``-1`` unknown, ``0`` CN, ``1`` MCI, ``2`` AD/Dementia,
        ``3`` FTD, ``4`` GENFI carrier.
    """
    if dx == "":
        dx = -1
    elif dx == "-1":
        dx = -1
    elif dx in ["CN", "Control", "CTR"]:
        dx = 0
    elif dx == "MCI":
        dx = 1
    elif dx in ["Dementia", "AD"]:
        dx = 2
    elif dx in ["FTD", "DFT"]:
        dx = 3
    elif dx in ["PortadorGENFI"]:
        dx = 4

    return dx


def get_dx_long(dx_list: list[str]) -> int:
    """Derive a longitudinal diagnosis code from a list of cross-sectional diagnoses.

    Combines baseline and last diagnosis to classify trajectories such as
    CN-stable, CN-to-MCI converter, MCI-to-AD converter, etc.

    Parameters
    ----------
    dx_list : list of str
        Ordered list of cross-sectional diagnosis labels (one per visit),
        as accepted by :func:`get_dx`.

    Returns
    -------
    int
        Longitudinal diagnosis code as defined in :data:`DX_DICT`.
    """
    if len(dx_list) == 1:
        return get_dx(dx_list[0])

    dx_bl = get_dx(dx_list[0])
    dx_last = get_dx(dx_list[-1])

    if dx_bl == -1 or dx_last == -1:
        dx_flag = [get_dx(d) != -1 for d in dx_list]

        if sum(dx_flag) == 0:
            return -1

        elif sum(dx_flag) == 1:
            idx = int(np.where(np.array(dx_flag) == 1)[0])
            return get_dx(dx_list[idx])

        else:
            idx = np.where(np.array(dx_flag) == 1)[0]
            idx_min = int(np.min(idx))
            idx_max = int(np.min(idx))
            dx_bl = get_dx(dx_list[idx_min])
            dx_last = get_dx(dx_list[idx_max])

    if dx_bl == 0:
        if dx_last == 0:
            dx = 3  # CN stable
        elif dx_last == 1:
            dx = 4  # CN-MCI converter
        elif dx_last == 2:
            dx = 5  # CN-AD converter
        elif dx_last == 3:
            dx = 11
        else:
            dx = 9
    elif dx_bl == 1:
        if dx_last == 1:
            dx = 6  # MCI_stable
        elif dx_last == 2:
            dx = 7  # MCI-AD converter
        else:
            dx = 9
    elif dx_bl == 2:
        if dx_last == 2:
            dx = 8  # AD stable
        else:
            dx = 9
    elif dx_bl == 3:
        if dx_last == 3:
            dx = 10  # AD stable
        else:
            dx = 9
    elif dx_bl == 4:
        if dx_last == 4:
            dx = 12  # GENFI
        else:
            dx = 9
    else:
        dx = 9  # Reversed diagnosis

    return dx


DX_DICT = {
    -1: "Unknown",
    0: "CN_cross",
    1: "MCI_cross",
    2: "Dementia_cross",
    3: "CN_stable",
    4: "CN_MCI_converter",
    5: "CN_AD_converter",
    6: "MCI_stable",
    7: "MCI_AD_converter",
    8: "AD_stable",
    9: "Reversed_diagnosis/Unknown",
    10: "FTD_stable",
    11: "FTD_converter",
    12: "GENFI",
}


def read_FS_volumes(file: str, labs: Optional[dict] = None) -> tuple[dict, float]:
    """Parse regional volumes and eTIV from a FreeSurfer stats file.

    Parameters
    ----------
    file : str
        Path to a FreeSurfer ``aseg.stats`` (or similar) file.
    labs : dict, optional
        Reserved for future use. Currently unused.

    Returns
    -------
    fs_vols : dict
        Mapping of structure name → volume (mm³) as a ``float``.
    etiv : float
        Estimated total intra-cranial volume in mm³.
    """
    etiv = 0
    fs_vols = {}
    start = False
    with open(file, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ")
        for row in csvreader:
            row_cool = list(filter(lambda x: x != "", row))
            if start is True:
                fs_vols[row_cool[1]] = float(row_cool[3])

            if "ColHeaders" in row_cool and start is False:
                start = True
            elif "EstimatedTotalIntraCranialVol," in row_cool:
                etiv = float(row_cool[-2].split(",")[0])

    # vols = {**{lab: float(fs_vols[lab_str]) for lab, lab_str in labs.items() if 'Thalamus' not in lab_str},
    #         **{lab: float(fs_vols[lab_str + '-Proper']) for lab, lab_str in labs.items() if 'Thalamus' in lab_str}}

    return fs_vols, etiv


def get_float_vol(v: Any) -> float:
    """Safely cast a value to float, returning ``-4`` on failure.

    Parameters
    ----------
    v : any
        Value to cast.

    Returns
    -------
    float
        Parsed float, or ``-4.0`` if conversion raises an exception.
    """
    try:
        return float(v)
    except Exception:
        return -4
