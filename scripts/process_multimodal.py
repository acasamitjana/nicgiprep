"""
Test script for the multimodal preprocessing pipeline.

Usage (from the repo root, with all env vars set):
    python scripts/process_multimodal.py \
        --bids data/data_tutorial/rawdata \
        --subjects ADNI002S1155 \
        --sessions m060 m072 m126

Required environment variables:
    BIDS_DIR            path to the rawdata folder (can also be passed as --bids)
    PYTHONPATH          repo root (e.g. export PYTHONPATH=/path/to/nicgiprep)
    FREESURFER_HOME     or FREESURFER_SYNTHMORPH_HOME  (for SynthSeg / SynthMorph)
"""

import os
import bids
from os.path import exists, join, dirname
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from setup import *
from nicgiprep.pipelines import (
    MultiModalSynthSegProcessor,
    MultiModalBiasCorrectionProcessor,
    MultiMRIProcessor,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = ArgumentParser(description="Multimodal preprocessing pipeline.")
parser.add_argument("--bids", default=BIDS_DIR, help="BIDS rawdata root directory")
parser.add_argument(
    "--subjects",
    default=None,
    nargs="+",
    help="Subject IDs to process (without sub- prefix)",
)
parser.add_argument(
    "--sessions",
    default=None,
    nargs="+",
    help="Session IDs to process (without ses- prefix); "
    "if omitted, all sessions are processed",
)
parser.add_argument(
    "--force", action="store_true", help="Rerun and overwrite existing outputs"
)
parser.add_argument(
    "--gpu", action="store_true", help="Enable GPU for SynthSeg (default: CPU)"
)
parser.add_argument(
    "--num_cores",
    default=1,
    type=int,
    help="Number of parallel worker threads for bias correction "
    "and registration (default: 1 = sequential). "
    "SynthSeg uses this value as its --threads count.",
)
args = parser.parse_args()

# ── BIDS layout ───────────────────────────────────────────────────────────────
print("Loading BIDS dataset...", end=" ", flush=True)
db_file = join(dirname(args.bids), "BIDS-raw.db")
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(
        root=args.bids, validate=False, database_path=db_file
    )

# Register all derivative directories needed by the multimodal pipeline.
for key in ("nicgiprep-cross", "nicgiprep-mm"):
    bids_loader.add_derivatives(DIR_PIPELINES[key])

print("done.")

# ── Step 1a: SynthSeg for non-T1w modalities (T1w done by cross-sectional pipeline) ──
print("\n=== Step 1a: SynthSeg for non-T1w modalities ===")
seg_processor = MultiModalSynthSegProcessor(
    bids_loader=bids_loader, subject_list=args.subjects
)
seg_processor.process(
    prefix="mm",
    gpu_flag=args.gpu,
    force_flag=args.force,
    threads=args.num_cores,
    session_list=args.sessions,
)

# ── Step 1b: bias-field correction for all modalities ────────────────────────
print("\n=== Step 1b: Bias field correction ===")
bias_processor = MultiModalBiasCorrectionProcessor(
    bids_loader=seg_processor.bids_loader, subject_list=args.subjects
)

if args.num_cores > 1:
    bias_processor.process_parallel(
        num_cores=args.num_cores, force_flag=args.force, session_list=args.sessions
    )
    bias_processor._update_full_layout()
else:
    subject_list = args.subjects or bias_processor.bids_loader.get_subjects()
    for subject in subject_list:
        bias_processor.process_subject(
            subject, force_flag=args.force, session_list=args.sessions
        )
        bias_processor._update_full_layout()

# ── Steps 2 & 3: joint registration → session space → MNI ────────────────────
print("\n=== Steps 2 & 3: Joint multimodal registration + MNI ===")
mm_processor = MultiMRIProcessor(
    bids_loader=bias_processor.bids_loader, subject_list=args.subjects
)

if args.num_cores > 1:
    mm_processor.process_parallel(
        num_cores=args.num_cores, force_flag=args.force, session_list=args.sessions
    )
    mm_processor._update_full_layout()
else:
    subject_list = args.subjects or mm_processor.bids_loader.get_subjects()
    for subject in subject_list:
        mm_processor.process_subject(
            subject, force_flag=args.force, session_list=args.sessions
        )
        mm_processor._update_full_layout()
