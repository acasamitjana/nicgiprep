"""Legacy setuptools shim — all packaging metadata lives in pyproject.toml.

The former pipeline configuration module is now ``nicgiprep/config.py``
(``from nicgiprep.config import *``). This file can be deleted.
"""
from setuptools import setup

setup()
