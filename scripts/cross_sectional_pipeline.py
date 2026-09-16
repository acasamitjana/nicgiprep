"""Backwards-compatible wrapper — the implementation lives in
:mod:`nicgiprep.scripts.cross_sectional_pipeline` and is installed as the ``nicgiprep-cross`` command.
"""
from nicgiprep.scripts.cross_sectional_pipeline import main

if __name__ == "__main__":
    main()
