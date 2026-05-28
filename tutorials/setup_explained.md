# `setup.py` — What it does

`setup.py` is **not** a packaging file. It is a shared configuration module imported by every script in `scripts/` via `from setup import *`. Its job is to read environment variables, build all the path constants the pipeline needs, and create output directories on disk.

---

## 1  Backend configuration

```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('NEURITE_BACKEND', 'tensorflow')
```

Silences TensorFlow C++ warnings and sets the backend for the `neurite` library (used by SynthMorph) to TensorFlow unless already overridden.

---

## 2  BIDS filename entities and path patterns

```python
filename_entities = ['subject', 'session', 'run', 'acquisition', ...]
BIDS_PATH_PATTERN = [...]
```

### `filename_entities`
A whitelist of BIDS key names that are allowed to appear in output filenames. Used throughout the pipeline when converting a `BIDSFile.entities` dict into a filename — any key not in this list is stripped before calling `build_path()`.

### `BIDS_PATH_PATTERN`
A list of four PyBIDS path-pattern strings, one per imaging modality (`anat`, `func`, `pet`, and a second `anat` pattern for scalar/transform files like affines and v2r matrices). The pipeline calls `bids_loader.build_path(entities, path_patterns=BIDS_PATH_PATTERN)` to construct output file paths in a BIDS-compliant way. Each pattern encodes:
- Mandatory entities (`subject`, `suffix`, `extension`)
- Optional entities in square brackets (`session`, `run`, `space`, …)
- Allowed values for constrained fields via `<opt1|opt2>` syntax

---

## 3  Repository root and data paths

```python
repo_home = os.environ.get('PYTHONPATH')
```

Reads `PYTHONPATH` as the repository root. This is expected to point to the `nicgiprep/` directory so that data files under `data/` can be located with absolute paths:

| Constant | Path | Purpose |
|----------|------|---------|
| `labels_registration` | `data/labels_classes_priors/label_list_registration.npy` | Brain-structure labels used for centroid-based rigid registration |
| `MNI_TEMPLATE` | `data/atlas/mni_icbm152_t1norm_*.nii.gz` | MNI152 T1w atlas |
| `MNI_TEMPLATE_SEG` | `data/atlas/mni_icbm152_synthseg_*.nii.gz` | SynthSeg parcellation of the MNI atlas |
| `MNI_TEMPLATE_MASK` | `data/atlas/mni_icbm152_mask_*.nii.gz` | Brain mask of the MNI atlas |
| `MNI_SM_V2R` | `data/atlas/mni_to_synthmorph_space.v2r.npy` | Vox-to-RAS for SynthMorph network space aligned to MNI |
| `MNI_ATLAS_TEMPLATE` / `_SEG` | `data/atlas/mni_reg_to_synthmorph_atlas.*` | MNI atlas registered into SynthMorph network space |

---

## 4  Dataset directories

### Required environment variable

```python
BIDS_DIR = os.environ['BIDS_DIR']   # e.g. /data/project/rawdata
```

The pipeline expects `BIDS_DIR` to point to the **rawdata** root of a BIDS dataset.  
`ROOT_DIR` is derived as the parent of `BIDS_DIR` (i.e. the dataset root).

### Optional environment variable

```python
DERIVATIVES_DIR = os.environ.get('DERIVATIVES_DIR', ROOT_DIR + '/derivatives')
```

Where pipeline outputs are written. Defaults to `<ROOT_DIR>/derivatives/` if not set explicitly.

### Scratch directories (always relative to `ROOT_DIR`)

| Constant | Path | Purpose |
|----------|------|---------|
| `LOGS_DIR` | `<ROOT_DIR>/logs/` | Log files |
| `TMP_DIR`  | `<ROOT_DIR>/tmp/`  | Intermediate files (file lists, checkpoints, temporary templates) |

All three directories are **created immediately** if they do not exist.

---

## 5  Pipeline output directories

```python
DIR_PIPELINES = {
    'preproc':   DERIVATIVES_DIR + '/preproc',
    'uslr-lin':  DERIVATIVES_DIR + '/uslr-lin',
    'uslr':      DERIVATIVES_DIR + '/uslr',
    'uslr-mni':  DERIVATIVES_DIR + '/uslr-mni',
}
```

Each entry corresponds to one stage of the USLR pipeline:

| Key | Produced by | Contents |
|-----|-------------|----------|
| `preproc` | `scripts/preprocess.py` | SynthSeg segmentations, bias-corrected T1w images, brain masks |
| `uslr-lin` | `scripts/linear_registration.py` | Per-session rigid affines, images/segs in subject space, subject template, eTIV |
| `uslr` | `scripts/nonlinear_registration.py` | Per-session SVFs, images/segs after deformable registration, nonlinear template |
| `uslr-mni` | any script with `--reg_MNI` | MNI-space images, segmentations, and affines |

Every directory is **created on import** and receives a `dataset_description.json` (BIDS-compliant metadata) the first time it is created.

---

## 6  First-run banner and FreeSurfer check

```python
if 'USLR_RUNNING' not in os.environ:
    # ... print ASCII logo ...
    if 'FREESURFER_HOME' not in os.environ:
        exit()
    os.environ['USLR_RUNNING'] = 'True'
```

When imported for the first time in a process (i.e. `USLR_RUNNING` is not yet set), `setup.py`:
1. Clears the terminal and prints the USLR ASCII logo.
2. Checks for a FreeSurfer installation (`FREESURFER_HOME` or `FREESURFER_SYNTHMORPH_HOME`). **Exits if neither is found.**
3. Prints the dataset and derivatives paths being used.
4. Sets `USLR_RUNNING=True` so subsequent imports within the same process skip the banner.

> **In notebooks**, set `os.environ['USLR_RUNNING'] = 'True'` **before** `from setup import *` to suppress the banner and the FreeSurfer exit check.

---

## 7  Summary of exported names

After `from setup import *`, every script/notebook has access to:

| Name | Type | Description |
|------|------|-------------|
| `BIDS_DIR` | `str` | Path to rawdata |
| `ROOT_DIR` | `str` | Dataset root (parent of rawdata) |
| `DERIVATIVES_DIR` | `str` | Root of all derivative outputs |
| `TMP_DIR` | `str` | Scratch space |
| `DIR_PIPELINES` | `dict` | Per-stage output directories |
| `BIDS_PATH_PATTERN` | `list[str]` | PyBIDS path patterns for `build_path()` |
| `filename_entities` | `list[str]` | Allowed BIDS entity keys for filenames |
| `labels_registration` | `str` | Path to label list `.npy` |
| `MNI_TEMPLATE` | `str` | Path to MNI T1w atlas |
| `MNI_TEMPLATE_SEG` | `str` | Path to MNI atlas segmentation |
| `MNI_TEMPLATE_MASK` | `str` | Path to MNI atlas brain mask |
