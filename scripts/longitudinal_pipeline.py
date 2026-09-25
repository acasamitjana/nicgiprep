"""Backwards-compatible wrapper — the implementation lives in
:mod:`nicgiprep.scripts.longitudinal_pipeline` and is installed as the ``nicgiprep-long`` command.
"""
from nicgiprep.scripts.longitudinal_pipeline import main

if __name__ == "__main__":
    main()
