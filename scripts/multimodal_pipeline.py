"""Backwards-compatible wrapper — the implementation lives in
:mod:`nicgiprep.scripts.multimodal_pipeline` and is installed as the ``nicgiprep-mm`` command.
"""
from nicgiprep.scripts.multimodal_pipeline import main

if __name__ == "__main__":
    main()
