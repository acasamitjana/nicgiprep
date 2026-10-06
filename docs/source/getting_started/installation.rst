Installation
============

Requirements
------------

* Python 3.10+
* FreeSurfer (with SynthSeg / SynthMorph support)
* A BIDS-formatted dataset

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/acasamitjana/nicgiprep.git
   cd nicgiprep
   pip install -e .            # or: pip install -e ".[dev,tutorials]"

All dependencies are declared in ``pyproject.toml``. The editable install is
required for now: the static resources (atlases, label lists) are read from the
repository ``data/`` folder. ``PYTHONPATH`` does not need to be set.

The install also provides three commands: ``nicgiprep-cross``,
``nicgiprep-long`` and ``nicgiprep-mm``.

Environment variables
---------------------

The following environment variables must be set before running any pipeline:

.. code-block:: bash

   export BIDS_DIR=/path/to/your/rawdata        # or pass --bids to the commands
   export FREESURFER_HOME=/path/to/freesurfer
   # optional
   export DERIVATIVES_DIR=/path/to/derivatives  # default: <BIDS_DIR>/../derivatives
