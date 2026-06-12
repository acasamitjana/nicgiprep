"""Brain segmentation label dictionaries and lookup tables.

This module loads SynthSeg, FreeSurfer aseg, and cortical parcellation labels
from the repository ``data/`` directory and exposes them as module-level
constants.

Constants
---------
SUPERSYNTH_LUT : dict
    Contiguous index LUT for SuperSYNTH labels.
SYNTHSEG_LUT : dict
    Contiguous index LUT for SynthSeg labels. Cortical labels are mapped to
    the index of label 3 (left GM) or label 42 (right GM).
SYNTHSEG_GMM_ONTOLOGY : dict
    Grouping of SynthSeg labels into named anatomical clusters
    (e.g. ``'Gray'``, ``'CSF'``, ``'WM'``).
CSF_LABELS : list of int
    Integer labels considered as CSF (currently ``[24]``).
"""

import numpy as np
from importlib.resources import files

# --- SuperSYNTH --- #
path = files("nicgiprep.data.labels_classes_priors").joinpath(
    "supersynth_segmentation_labels.npy"
)
supersynth_labels = np.load(path)
SUPERSYNTH_LUT = {k: it_k for it_k, k in enumerate(np.unique(supersynth_labels))}

# --- SynthSeg --- #
path = files("nicgiprep.data.labels_classes_priors").joinpath(
    "synthseg_parcellation_labels.npy"
)
ctx_labels = np.load(path)

path = files("nicgiprep.data.labels_classes_priors").joinpath(
    "synthseg_segmentation_labels.npy"
)
subcortical_labels = np.load(path)
subcortical_labels = np.concatenate((subcortical_labels, [24]))

SYNTHSEG_LUT = {k: it_k for it_k, k in enumerate(np.unique(subcortical_labels))}
SYNTHSEG_LUT = {
    **SYNTHSEG_LUT,
    **{
        k: SYNTHSEG_LUT[3] if k < 2000 else SYNTHSEG_LUT[42]
        for k in ctx_labels
        if k != 0
    },
}

SYNTHSEG_APARC_LUT = {
    k: it_k
    for it_k, k in enumerate(
        np.unique(np.concatenate((subcortical_labels, ctx_labels), axis=0))
    )
}

SYNTHSEG_GMM_ONTOLOGY = {
    "Gray": [53, 17, 51, 12, 54, 18, 50, 11, 58, 26, 42, 3],
    "CSF": [4, 5, 43, 44, 15, 14, 24],
    "Thalaumus": [49, 10],
    "Pallidum": [52, 13],
    "VentralDC": [28, 60],
    "Brainstem": [16],
    "WM": [41, 2],
    "cllGM": [47, 8],
    "cllWM": [46, 7],
}

CSF_LABELS = [24]

path = files("nicgiprep.data.labels_classes_priors").joinpath("labels_registration.npy")
labels_registration = np.load(path)
