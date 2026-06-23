
"""
Script for the cross-sectional preprocessing pipeline.

Example Usage (from the repo root, with all env vars set):
    python scripts/process_cross.py \
        --bids data/data_tutorial/rawdata \
        --subjects ADNI002S1155 \
        --sessions m060 m072 m126
        --gpu
        --gpu_num 0
        --num_cores 64

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

from nicgiprep.pipelines import (SynthSegProcessor, 
                                 BiasCorrectionProcessor, 
                                 MNIRegistrationProcessor)


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = ArgumentParser(description='Cross-sectional preprocessing pipeline.')
parser.add_argument('--bids',
                    default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
parser.add_argument('--subjects',
                    default=None, nargs='+', help="(optional) specify which subjects to process")
parser.add_argument("--sessions",
                    default=None, nargs="+", help="Session IDs to process (without ses- prefix); "
                    "if omitted, all sessions are processed")
parser.add_argument("--force", action='store_true',
                    help="Rerun and overwrite existing outputs.")
parser.add_argument("--gpu", action="store_true", help="Enable GPU for SuperSynth/SynthSeg (default: GPU/CPU)")
parser.add_argument("--gpu_num", default=0, help="GPU number to be used for SuperSynth/SynthSeg")
parser.add_argument('--num_cores',
                    default=1, type=int, help="number of parallel cores to run.")
args = parser.parse_args()


# ── BIDS layout ───────────────────────────────────────────────────────────────
print('LOADING Dataset ...', end=' ', flush=True)
db_file = join(dirname(args.bids), 'BIDS-raw.db')
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False, database_path=db_file)

bids_loader.add_derivatives(DIR_PIPELINES['nicgiprep-cross'])

if args.subjects is None:
    print('Total subjects found: N=' + str(len(bids_loader.get_subjects())), end=' ', flush=True)
else:
    print('Total subjects found: N=' + str(len(args.subjects)), end=' ', flush=True)


# ── GPU init ───────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)


# ── Step 1: SynthSeg and SuperSynth for T1w modality ──────────────────────
print("\n=== Step 1: SynthSeg and SuperSynth for T1w modality ===")
processing_seg = SynthSegProcessor(
    bids_loader=bids_loader,
    subject_list=args.subjects
    )

processing_seg.process(
    force_flag=args.force,
    gpu_flag=args.gpu,
    threads=args.num_cores,
    session_list=args.sessions,
    )


# ── Step 2: bias-field correction for T1w modality ────────────────────────
print("\n=== Step 2: bias-field correction for T1w modality ===")
processing_bias = BiasCorrectionProcessor(
    bids_loader=bids_loader,
    subject_list=args.subjects
    )

processing_bias.process(force_flag=args.force, session_list=args.sessions)


# ── Step 3: MNI Registration for T1w modality ─────────────────────────────
print("\n=== Step 3: MNI Registration for T1w modality ===")

processing_mni = MNIRegistrationProcessor(
    bids_loader=bids_loader,
    subject_list=args.subjects,
    )

processing_mni.process(force_flag=args.force, session_list=args.sessions)

