import os
import json
import pdb

import subprocess
from importlib.resources import files


# ------------- #
# BIDS features #
# ------------- #
filename_entities = [
    "subject",
    "session",
    "run",
    "acquisition",
    "suffix",
    "extension",
    "task",
    "tracer",
    "reconstruction",
    "desc",
    "space",
    "datatype",
]
BIDS_PATH_PATTERN = [
    "sub-{subject}[/ses-{session}]/{datatype<anat|utils>|anat}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<aff|v2r|etiv|T1wetiv>}{extension<.txt|.npy>|.npy}",
    "sub-{subject}[/ses-{session}]/{datatype<anat|utils>|anat}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<T1w|T2w|T2star|T2starw|FLAIR|FLASH|PD|PDw|PDT2|inplaneT[12]|angio|dseg|posteriors|svf|jac|def|T1wpost|T1wstats|T1wstd|T1wmask|T1wdseg|T2wmask|T2wdseg|FLAIRmask|FLAIRdseg|PDwmask|PDwdseg|PDmask|PDdseg|T1wsynthseg|T2wsynthseg|FLAIRsynthseg|PDwsynthseg|mask|space>}{extension<.nii|.nii.gz|.json|.txt|.npy>|.nii.gz}",
    "sub-{subject}[/ses-{session}]/{datatype<func>|func}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<bold|cbv|sbref>}{extension<.nii|.nii.gz|.json|.txt|.npy>|.nii.gz}",
    "sub-{subject}[/ses-{session}]/{datatype<pet>|pet}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_trc-{tracer}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<pet>}{extension<.nii|.nii.gz|.json|.txt|.npy>|.nii.gz}",
    "sub-{subject}[/ses-{session}]/{datatype<utils>|utils}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<svf|aff|empty|v2r|T1wsynthseg>}{extension<.nii|.nii.gz|.npy>|.nii.gz}",
]

# Custom pybids config: extends the default "bids" config so BIDSLayout.
# Pass config=[str(BIDS_CONFIG), "derivatives"] to BIDSLayout/add_derivatives
# to use it in place of the built-in "bids" config.
BIDS_CONFIG = files("data.config").joinpath("nicgiprep_bids.json")


# ------------- #
# MNI templates #
# ------------- #

# MNI Templates
MNI_TEMPLATE = files("data.atlas").joinpath(  #HACK: removed nicgiprep.
    "mni_icbm152_t1norm_tal_nlin_sym_09a.nii.gz"
)
MNI_SM_V2R = files("data.atlas").joinpath(  # HACK: removed nicgiprep.
    "mni_icbm152_t1norm_tal_nlin_sym_09a.nii.gz"
)
MNI_ATLAS_TEMPLATE = files("data.atlas").joinpath(  # HACK: removed nicgiprep.
    "mni_reg_to_synthmorph_atlas.nii.gz"
)
MNI_ATLAS_TEMPLATE_SEG = files("data.atlas").joinpath(  # HACK: removed nicgiprep.
    "mni_reg_to_synthmorph_atlas.seg.nii.gz"
)
MNI_ATLAS_TEMPLATE_MASK = files("data.atlas").joinpath(  # HACK: removed nicgiprep.
    "mni_reg_to_synthmorph_atlas.mask.nii.gz"
)
MNI_TEMPLATE_SEG = files("data.atlas").joinpath(  # HACK: removed nicgiprep.
    "mni_icbm152_synthseg_tal_nlin_sym_09a.nii.gz"
)
MNI_TEMPLATE_MASK = files("data.atlas").joinpath(  # HACK: removed nicgiprep.
    "mni_icbm152_mask_tal_nlin_sym_09a.nii.gz"
)


# ------------ #
# TF variables #
# ------------ #
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "NEURITE_BACKEND" not in os.environ:
    os.environ["NEURITE_BACKEND"] = "tensorflow"


# ---------------------- #
# Project data structure #
# ---------------------- #
BIDS_DIR = os.environ.get("BIDS_DIR")
if not BIDS_DIR:
    raise ValueError("Please, specify --bids or set the BIDS_DIR environment variable.")
BIDS_DIR = BIDS_DIR.rstrip("/")
os.environ["BIDS_DIR"] = BIDS_DIR
ROOT_DIR = os.path.dirname(BIDS_DIR)

DERIVATIVES_DIR = os.environ.get("DERIVATIVES_DIR")
if DERIVATIVES_DIR:
    DERIVATIVES_DIR = DERIVATIVES_DIR.rstrip("/")
else:
    DERIVATIVES_DIR = os.path.join(ROOT_DIR, "derivatives")
os.environ["DERIVATIVES_DIR"] = DERIVATIVES_DIR

LOGS_DIR = os.path.join(ROOT_DIR, "logs")
TMP_DIR = os.path.join(ROOT_DIR, "tmp")

os.makedirs(DERIVATIVES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

DIR_PIPELINES = {
    "nicgiprep-cross": os.path.join(DERIVATIVES_DIR, "nicgiprep-cross"),
    'nicgiprep-long': os.path.join(DERIVATIVES_DIR, 'nicgiprep-long'),
    "nicgiprep-mm": os.path.join(DERIVATIVES_DIR, "nicgiprep-mm"),
}
DESC_PIPELINES = {
    'nicgiprep-cross': 'Cross-sectional preprocessing pipeline outcomes',
    'nicgiprep-long': 'Longitudinal preprocessing pipeline outcomes',
    'nicgiprep-mm': 'Multimodal preprocessing pipeline outcomes',
}


for d, d_str in DESC_PIPELINES.items():
    os.makedirs(DIR_PIPELINES[d], exist_ok=True)
    data_descr_path = os.path.join(DIR_PIPELINES[d], "dataset_description.json")
    if not os.path.exists(data_descr_path):
        data_descr = {
            "Name": os.path.basename(d_str),
            "BIDSVersion": "1.0.2",
            "GeneratedBy": [{"Name": d}],
            "Description": d_str,
        }
        with open(data_descr_path, "w") as outfile:
            json.dump(data_descr, outfile, indent=4)



# --------------------- #
# Initialize sys.stdout #
# --------------------- #
os.system("cls" if os.name == "nt" else "clear")
print("            oo            ")
print("          oooooo          ")
print("        oooooooooo        ")
print("      oooooooooooooo      ")
print("    oooooooooooooooooo    ")
print("  oooooooooooooooooooooo  ")
print("oooooooooooooooooooooooooo")
print("")
print("Running NicGiPrep Pipeline")
print("")
print("oooooooooooooooooooooooooo")
print("oooooooooooooooooooooooooo")
print("")

print("* RAWDATA DIRECTORY ($BIDS_DIR or --bids):      " + BIDS_DIR)
print("* DERIVATIVES DIRECTORY:                        " + DERIVATIVES_DIR)

if "FREESURFER_HOME" in os.environ:
    subprocess.call(["bash", "-c", "source $FREESURFER_HOME/SetUpFreeSurfer.sh"])
    print("* FREESURFER HOME:                              " + os.environ["FREESURFER_HOME"])

else:
    print("Please, source FREESURFER first for registration and segmentation.")
    exit()



print("")
print("oooooooooooooooooooooooooo")
print("oooooooooooooooooooooooooo")
print("\n")
