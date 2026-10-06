# `nicgiprep/config.py` and `nicgiprep/resources.py` — what they do

The old root-level `setup.py` (imported with `from setup import *`) has been split in two modules inside the package. `setup.py` at the repository root is now only a setuptools shim; all packaging metadata lives in `pyproject.toml`.

| Module | Side effects on import | Use it for |
|--------|------------------------|------------|
| `nicgiprep.resources` | none | paths to atlases, label lists and the pybids config |
| `nicgiprep.config` | reads env vars, creates folders, prints banner, checks FreeSurfer | everything the pipelines need at run time |

Install the package once with `pip install -e .` from the repository root. `PYTHONPATH` is no longer needed.

---

## 1  `nicgiprep.resources` — static resources

The data folder is located automatically, in this order:

1. `$NICGIPREP_DATA_DIR` (if set)
2. `nicgiprep/data/` (if the data folder is moved inside the package)
3. `<repo>/data/` (current layout — works with editable installs)

| Constant | Path | Purpose |
|----------|------|---------|
| `DATA_DIR` | `data/` | Root of the resources |
| `BIDS_CONFIG` | `data/config/nicgiprep_bids.json` | Custom pybids config (`config=[str(BIDS_CONFIG), "derivatives"]`) |
| `MNI_TEMPLATE` | `data/atlas/mni_icbm152_t1norm_*.nii.gz` | MNI152 T1w atlas |
| `MNI_TEMPLATE_SEG` | `data/atlas/mni_icbm152_synthseg_*.nii.gz` | SynthSeg parcellation of the MNI atlas |
| `MNI_TEMPLATE_MASK` | `data/atlas/mni_icbm152_mask_*.nii.gz` | Brain mask of the MNI atlas |
| `MNI_SM_V2R` | see note in the module | Vox-to-RAS for SynthMorph space aligned to MNI |
| `MNI_ATLAS_TEMPLATE` / `_SEG` / `_MASK` | `data/atlas/mni_reg_to_synthmorph_atlas.*` | MNI atlas in SynthMorph space |

Label lists (`labels_registration`, SynthSeg / SuperSynth LUTs) are exposed by `nicgiprep.utils.label_utils`.

---

## 2  `nicgiprep.config` — run-time configuration

### Environment variables

| Variable | Required | Meaning |
|----------|----------|---------|
| `BIDS_DIR` | yes (or `--bids`) | rawdata root of the BIDS dataset |
| `DERIVATIVES_DIR` | no | defaults to `<ROOT_DIR>/derivatives` |
| `FREESURFER_HOME` | yes | FreeSurfer installation (SynthSeg / SynthMorph). The process exits if it is not set. |

Set them **before** importing `nicgiprep.config` or any `nicgiprep.pipelines` module (the command-line tools do this for you from `--bids` / `--derivatives`).

### Constants

| Name | Description |
|------|-------------|
| `BIDS_DIR` | Path to rawdata |
| `ROOT_DIR` | Dataset root (parent of rawdata) |
| `DERIVATIVES_DIR` | Root of all derivative outputs |
| `LOGS_DIR`, `TMP_DIR` | `<ROOT_DIR>/logs`, `<ROOT_DIR>/tmp` |
| `DIR_PIPELINES` | Output folder per pipeline: `nicgiprep-cross`, `nicgiprep-long`, `nicgiprep-mm` |
| `DESC_PIPELINES` | Description written to each `dataset_description.json` |
| `BIDS_PATH_PATTERN` | PyBIDS path patterns for `build_path()` |
| `filename_entities` | BIDS entity keys allowed in output filenames |
| all `nicgiprep.resources` constants | re-exported |

On import, the derivatives, logs and tmp folders are created, and each pipeline folder gets a `dataset_description.json` the first time.

---

## 3  Command-line tools

`pip install -e .` installs:

| Command | Module |
|---------|--------|
| `nicgiprep-cross` | `nicgiprep/scripts/cross_sectional_pipeline.py` |
| `nicgiprep-long` | `nicgiprep/scripts/longitudinal_pipeline.py` |
| `nicgiprep-mm` | `nicgiprep/scripts/multimodal_pipeline.py` |

The files in the top-level `scripts/` folder are thin wrappers around these modules.

---

## 4  In notebooks

```python
import os
os.environ["BIDS_DIR"] = "/path/to/rawdata"
os.environ["FREESURFER_HOME"] = "/path/to/freesurfer"

from nicgiprep.config import *
```
