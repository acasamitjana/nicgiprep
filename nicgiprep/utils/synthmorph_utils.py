import pdb
from typing import Union, Literal, Any
import os
from os.path import join, exists
from pathlib import Path

from bids.layout import BIDSFile
import tensorflow as tf
import h5py
import numpy as np
import neurite as ne
from voxelmorph import layers as vxm_layers
from voxelmorph import utils as vxm_utils
import voxelmorph as vxm
import nibabel as nib

SpatialTransformer = vxm_layers.SpatialTransformer
compose_transforms = vxm_utils.compose
constant_layer = ne.layers.Constant

def load_weights(model: tf.keras.Model, weights: str|Path) -> None:
    """Load weights into model or submodel.

    Attempts to load (all) weights into a model or one of its submodels. If
    that fails, `model` may be a submodel of what we got weights for, and we
    attempt to load the weights of a submodel (layer) into `model`.

    Parameters
    ----------
    model : TensorFlow (keras) model
        Model to initialize.
    weights : str or pathlib.Path
        Path to weights file.

    Raises
    ------
    ValueError
        If unsuccessful at loading any weights.

    """
    # Extract submodels.
    models = [model]
    i = 0
    while i < len(models):
        layers = [f for f in models[i].layers if isinstance(f, tf.keras.Model)]
        models.extend(layers)
        i += 1

    # Add models wrapping a single model in case this was done in training.
    # Requires list expansion or Python will get stuck.
    models.extend([tf.keras.Model(m.inputs, m(m.inputs)) for m in models])

    # Attempt to load all weights into one of the models.
    for mod in models:
        try:
            mod.load_weights(weights)
            return
        except ValueError as e:
            pass

    # Assume `model` is a submodel of what we got weights for.
    with h5py.File(weights, mode='r') as h5:
        layers = h5.attrs['layer_names']
        weights = [list(h5[lay].attrs['weight_names']) for lay in layers]

        # Layers with weights. Attempt loading.
        layers, weights = zip(*filter(lambda f: f[1], zip(layers, weights)))
        for lay, wei in zip(layers, weights):
            try:
                model.set_weights([h5[lay][w] for w in wei])
                return
            except ValueError as e:
                if lay is layers[-1]:
                    raise e

def warp(flo_arr: np.ndarray, def_arr: np.ndarray) -> np.ndarray:
    """Apply a dense displacement field to a 3D volume using TensorFlow.

    Parameters
    ----------
    flo_arr : np.ndarray
        Floating (moving) image, shape ``(X, Y, Z)`` or ``(X, Y, Z, C)``.
    def_arr : np.ndarray
        Dense displacement field, shape ``(X, Y, Z, 3)``.

    Returns
    -------
    np.ndarray
        Warped image with the same spatial shape as ``arr_def``.
    """
    if len(flo_arr.shape) == 3:
        flo_arr = flo_arr[..., np.newaxis]
        
    tf_flo_arr = tf.cast(flo_arr[np.newaxis], tf.float32)
    tf_def_arr = tf.cast(def_arr[np.newaxis], tf.float32)

    prop = dict(shift_center=False, fill_value=0, shape=tf_def_arr.shape[1:-1])
    mov_1 = vxm_layers.SpatialTransformer(**prop)((tf_flo_arr, tf_def_arr))

    return np.squeeze(np.array(mov_1))

def integrate_svf(svf: np.ndarray, orig_shape: tuple[int], scaling_factor: int=2, int_steps: int=7) -> np.ndarray:
    """Integrate a stationary velocity field (SVF) via scaling and squaring.

    Parameters
    ----------
    svf : np.ndarray
        Stationary velocity field, shape ``(X, Y, Z, 3)``.
    orig_shape : tuple of int
        Spatial shape of the target image space, used to compose a
        downsampling affine after integration.
    scaling_factor : int, optional
        Up-scaling factor applied before integration. Default is 2.
    int_steps : int, optional
        Number of scaling-and-squaring steps. Default is 7.

    Returns
    -------
    np.ndarray
        Dense displacement field, shape ``(X, Y, Z, 3)``.
    """
    num_dim = len(svf.shape) - 1

    def scale(fact: float) -> Any:
        """Return a constant isotropic scaling affine layer with factor ``fact``."""
        mat = np.diag((*[fact] * num_dim, 1))
        return ne.layers.Constant(mat)([])

    aff_1 = scale(scaling_factor)
    fw_def = vxm_layers.VecInt(method='ss', int_steps=int_steps)(svf[np.newaxis])
    fw_def = (aff_1, fw_def, scale(1 / scaling_factor), aff_1)
    fw_def = vxm_layers.ComposeTransform(shift_center=False)(fw_def)

    down = vxm_layers.AffineToDenseShift(orig_shape, shift_center=False)(scale(1 / scaling_factor))
    fw_def = vxm_layers.ComposeTransform()((fw_def, down))

    return np.squeeze(np.array(fw_def))


def synthmorph_register(imref_file: Union[str, nib.Nifti1Image, BIDSFile],
                        imflo_file: Union[str, nib.Nifti1Image, BIDSFile],
                        reg_param: Union[float, int] = 0.5) -> Union[np.ndarray, None]:
    """Register a floating image to a reference using SynthMorph deformable registration.

    Loads the SynthMorph model from
    ``$FREESURFER_HOME/models/synthmorph.deform.3.h5`` and returns the
    forward stationary velocity field (SVF).

    Parameters
    ----------
    imref_file : str, nibabel.Nifti1Image, or file-like
        Reference (fixed) image. Accepts a file path, a ``Nifti1Image``
        object, or an object with a ``.path`` attribute.
    imflo_file : str, nibabel.Nifti1Image, or file-like
        Floating (moving) image. Same format options as ``imageref_file``.
    reg_param : float or int, optional
        Regularisation hyper-parameter for the HyperVxmJoint model.
        Higher values produce smoother deformations. Default is 0.5.

    Returns
    -------
    np.ndarray
        Forward SVF in the reference image voxel space, shape ``(X, Y, Z, 3)``.
    """
    # subprocess.call(
    #     ['mri_synthmorph', 'register', '-m', 'deform', '-t', 'prova_sm_def.nii.gz', '-o', 'prova_sm_mov.nii.gz',
    #      imageflo_file.path,
    #      imageref_file.path], stdout=subprocess.DEVNULL)
    #
    # pdb.set_trace()

    if not exists(join(str(os.environ.get('FREESURFER_HOME')), 'models', 'synthmorph.deform.3.h5')):
        return None


    if isinstance(imref_file, nib.Nifti1Image):
        ref_proxy = imref_file
    elif isinstance(imref_file, str):
        ref_proxy = nib.load(imref_file)
    else:
        ref_proxy = nib.load(imref_file.path)

    if isinstance(imflo_file, nib.Nifti1Image):
        flo_proxy = imflo_file
    elif isinstance(imflo_file, str):
        flo_proxy = nib.load(imflo_file)
    else:
        flo_proxy = nib.load(imflo_file.path)

    ref_arr = np.array(ref_proxy.dataobj)
    flo_arr = np.array(flo_proxy.dataobj)

    prop = dict(in_shape=ref_arr.shape, bidir=True, int_steps=7, skip_affine=True, return_svf=True)
    model = vxm.networks.HyperVxmJoint(**prop)

    ref_arr -= np.min(ref_arr)
    ref_arr /= np.max(ref_arr)
    ref_arr = ref_arr[np.newaxis, ..., np.newaxis]

    flo_arr -= np.min(flo_arr)
    flo_arr /= np.max(flo_arr)
    flo_arr = flo_arr[np.newaxis, ..., np.newaxis]

    inputs = (tf.constant([reg_param]), tf.cast(flo_arr, tf.float32), tf.cast(ref_arr, tf.float32))
    load_weights(model, weights=join(str(os.environ.get('FREESURFER_HOME')), 'models', 'synthmorph.deform.3.h5'))

    _, _, fw_svf, _ = tuple(map(tf.squeeze, model(inputs)))

    return np.squeeze(np.array(fw_svf))
