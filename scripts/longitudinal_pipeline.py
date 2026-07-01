"""Command-line entry point for the nicgiprep longitudinal (USLR) pipeline.

Parses CLI arguments and an optional YAML configuration file, builds the
PyBIDS layout (adding the cross-sectional and longitudinal derivatives), and
runs :class:`~nicgiprep.pipelines.longitudinal.LongitudinalRegistration` over
the requested subjects.
"""

import os
from os.path import join, exists
from argparse import ArgumentParser

import yaml


def parse_args():
    """Parse command-line arguments for the longitudinal pipeline."""
    parser = ArgumentParser(description="Run the nicgiprep longitudinal (USLR) registration pipeline.")
    parser.add_argument("--bids", default=os.environ.get("BIDS_DIR"),
                         help="Path to the BIDS rawdata directory. Defaults to the BIDS_DIR environment variable.")
    parser.add_argument("--derivatives", default=None,
                         help="Root derivatives directory. Defaults to <bids_parent>/derivatives.")
    parser.add_argument("--cross-derivatives", default=None,
                         help="Path to the nicgiprep-cross derivatives directory. "
                              "Defaults to <derivatives>/nicgiprep-cross.")
    parser.add_argument("--subjects", default=None, nargs="+",
                         help="(optional) specify which subjects to process.")
    parser.add_argument("--config", default=None,
                         help="(optional) path to a YAML file with pipeline parameters (e.g. cost).")
    parser.add_argument("--cost", default=None, choices=["bch-l1", "bch-l2"],
                         help="Cost function for the nonlinear spanning-tree solve.")
    parser.add_argument("--force", action="store_true",
                         help="Reprocess subjects even if outputs already exist.")
    parser.add_argument("--verbose", action="store_true",
                         help="Print progress messages.")

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
    """Build the BIDS layout and run the longitudinal pipeline."""
    args = parse_args()
    config = load_config(args.config)

    # The environment must be configured before importing from `setup` or
    # `nicgiprep`, since `setup.py` reads BIDS_DIR/DERIVATIVES_DIR at import time.
    os.environ["BIDS_DIR"] = args.bids
    if args.derivatives is not None:
        os.environ["DERIVATIVES_DIR"] = args.derivatives

    import bids
    from setup import BIDS_DIR, ROOT_DIR, DERIVATIVES_DIR, DIR_PIPELINES
    from nicgiprep.pipelines.longitudinal import LongitudinalRegistration

    cross_derivatives = args.cross_derivatives or join(DERIVATIVES_DIR, "nicgiprep-cross")

    print("LOADING Dataset ...", end=" ", flush=True)
    db_file = join(ROOT_DIR, "BIDS-raw.db")
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=BIDS_DIR, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=BIDS_DIR, validate=False, database_path=db_file)

    bids_loader.add_derivatives(cross_derivatives)
    bids_loader.add_derivatives(DIR_PIPELINES["nicgiprep-long"])

    processing = LongitudinalRegistration(bids_loader=bids_loader, subject_list=args.subjects)

    print("Total subjects found: N=" + str(len(processing.subject_list)), end="\n\n")

    process_kwargs = {**config}
    if args.cost is not None:
        process_kwargs["cost"] = args.cost
    process_kwargs["force_flag"] = args.force
    process_kwargs["verbose"] = args.verbose

    processing.process(**process_kwargs)


if __name__ == "__main__":
    main()