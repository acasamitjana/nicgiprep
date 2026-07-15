"""
Script for the cross-sectional preprocessing pipeline.

Example Usage (from the repo root, with all env vars set):
    python scripts/cross_sectional_pipeline.py \
        --bids data/data_tutorial/rawdata \
        --subjects ADNI002S1155 \
        --sessions m060 m072 m126
        --gpu
        --gpu_num 0
        --num_cores 64

Required environment variables:
    BIDS_DIR                      path to the rawdata folder (can also be passed as --bids)
    DERIVATIVES_DIR (optional)    path to the derivatives folder (default: same level as BIDS_DIR)
    FREESURFER_HOME               (for segmentation and registration)
"""
import os
from os.path import exists, join, dirname
from argparse import ArgumentParser
import copy

import yaml


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = ArgumentParser(description="Run the NicGiPrep cross-sectional processing pipeline.")
    parser.add_argument("--bids", default=os.environ.get("BIDS_DIR"),
                         help="Path to the BIDS rawdata directory. Defaults to the BIDS_DIR environment variable.")
    parser.add_argument("--derivatives", default=None,
                         help="Root derivatives directory. Defaults to <bids_parent>/derivatives.")
    parser.add_argument("--subjects", default=None, nargs="+",
                         help="(optional) specify which subjects to process.")
    parser.add_argument("--sessions", default=None, nargs="+",
                        help="(optional) specify session IDs to process (without ses- prefix); "
                             "all sessions are processed by default")
    parser.add_argument("--config", default=None,
                        help="(optional) path to a YAML file with pipeline parameters (e.g. cost).")
    parser.add_argument("--force", action="store_true",
                         help="Reprocess subjects even if outputs already exist.")
    parser.add_argument("--verbose", action="store_true",
                         help="Print progress messages.")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU for SuperSynth/SynthSeg.")
    parser.add_argument("--gpu_num", default=0,
                        help="Specific GPU number to be used (default uses CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--num_cores", default=1, type=int,
                        help="number of parallel cores to run in SynthSeg and SuperSynth segmentation (default: 1).")

    args = parser.parse_args()
    if args.bids is None:
        parser.error("--bids must be specified (or set the BIDS_DIR environment variable).")

    return args


def load_config(config_path):
    """Load pipeline parameters from a YAML configuration file.

    Parameters
    ----------
    config_path : str or None
        Path to the YAML file. If ``None``, an empty dict is returned.

    Returns
    -------
    dict
        Parsed configuration, forwarded as keyword arguments to
        ``LongitudinalRegistration.process``.
    """
    if config_path is None:
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    config = load_config(args.config)

    # The environment must be configured before importing from `setup` or
    # `nicgiprep`, since `setup.py` reads BIDS_DIR/DERIVATIVES_DIR at import time.
    os.environ["BIDS_DIR"] = args.bids
    if args.derivatives is not None:
        os.environ["DERIVATIVES_DIR"] = args.derivatives

    # ── Imports ───────────────────────────────────────────────────────────────
    import bids
    from setup import BIDS_DIR, ROOT_DIR, DERIVATIVES_DIR, DIR_PIPELINES

    from nicgiprep.pipelines import (
        SynthSegProcessor,
        BiasCorrectionProcessor,
        MNIRegistrationProcessor,
    )

    # ── GPU init ───────────────────────────────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

    # ── BIDS layout ───────────────────────────────────────────────────────────────
    print("LOADING Dataset ...", end=" ", flush=True)
    db_file = join(ROOT_DIR, "BIDS-raw.db")
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=BIDS_DIR, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=BIDS_DIR, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES["nicgiprep-cross"])

    processing_seg = SynthSegProcessor(bids_loader=bids_loader, subject_list=args.subjects)

    print("Total subjects found: N=" + str(len(processing_seg.subject_list)), end="\n\n")


    process_kwargs = {**config}
    process_kwargs["force_flag"] = args.force
    process_kwargs["session_list"] = args.sessions

    proces_seg_kwargs = copy.deepcopy(process_kwargs)
    proces_seg_kwargs["gpu_flag"] = args.gpu
    proces_seg_kwargs["threads"] = args.num_cores
    proces_seg_kwargs["verbose"] = args.verbose


    # ── Step 1: SynthSeg and SuperSynth for T1w modality ──────────────────────
    print("\n=== Step 1: SynthSeg and SuperSynth for T1w modality ===")
    processing_seg.process(**process_kwargs)


    # ── Step 2: bias-field correction for T1w modality ────────────────────────
    print("\n=== Step 2: bias-field correction for T1w modality ===")
    processing_bias = BiasCorrectionProcessor(
        bids_loader=bids_loader, subject_list=args.subjects
    )

    processing_bias.process(force_flag=args.force, session_list=args.sessions)


    # ── Step 3: MNI Registration for T1w modality ─────────────────────────────
    print("\n=== Step 3: MNI Registration for T1w modality ===")

    processing_mni = MNIRegistrationProcessor(
        bids_loader=bids_loader,
        subject_list=args.subjects,
    )

    processing_mni.process(force_flag=args.force, session_list=args.sessions)
