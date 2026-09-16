"""Paths to the static resources shipped with NicGiPrep.

This module has **no side effects**: it only resolves file locations, so it
is safe to import from anywhere (utilities, tests, notebooks) without setting
``BIDS_DIR`` or sourcing FreeSurfer.

Resources are looked up in ``nicgiprep/data`` first (location used when the
data folder lives inside the package) and fall back to the repository-level
``data/`` folder (current layout, valid for editable installs
``pip install -e .``). The ``NICGIPREP_DATA_DIR`` environment variable
overrides both.
"""

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def _find_data_dir() -> Path:
    """Return the directory that holds the NicGiPrep static resources.

    Returns
    -------
    pathlib.Path
        First existing candidate among ``$NICGIPREP_DATA_DIR``,
        ``nicgiprep/data`` and ``<repo>/data``.

    Raises
    ------
    FileNotFoundError
        If none of the candidate directories exists.
    """
    candidates = []
    if os.environ.get("NICGIPREP_DATA_DIR"):
        candidates.append(Path(os.environ["NICGIPREP_DATA_DIR"]))
    candidates += [_PACKAGE_DIR / "data", _PACKAGE_DIR.parent / "data"]

    for candidate in candidates:
        if (candidate / "labels_classes_priors").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find the NicGiPrep data directory. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Install the package in editable mode (`pip install -e .`) from the "
        "repository root, or set NICGIPREP_DATA_DIR."
    )


DATA_DIR = _find_data_dir()
ATLAS_DIR = DATA_DIR / "atlas"
LABELS_DIR = DATA_DIR / "labels_classes_priors"
CONFIG_DIR = DATA_DIR / "config"

# Custom pybids config: extends the default "bids" config.
# Pass config=[str(BIDS_CONFIG), "derivatives"] to BIDSLayout/add_derivatives.
BIDS_CONFIG = CONFIG_DIR / "nicgiprep_bids.json"

# MNI templates
MNI_TEMPLATE = ATLAS_DIR / "mni_icbm152_t1norm_tal_nlin_sym_09a.nii.gz"
MNI_TEMPLATE_SEG = ATLAS_DIR / "mni_icbm152_synthseg_tal_nlin_sym_09a.nii.gz"
MNI_TEMPLATE_MASK = ATLAS_DIR / "mni_icbm152_mask_tal_nlin_sym_09a.nii.gz"
