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

   git clone https://github.com/<org>/nicgiprep.git
   cd nicgiprep
   pip install -e .

Environment variables
---------------------

The following environment variables must be set before running any pipeline:

.. code-block:: bash

   export BIDS_DIR=/path/to/your/rawdata
   export PYTHONPATH=/path/to/nicgiprep
   export FREESURFER_HOME=/path/to/freesurfer   # or FREESURFER_SYNTHMORPH_HOME
