"""
Base Processing pipeline schematic for neuroimage preprocessing.

Provides base classe that implements basic and necessary functions
"""

import copy
from joblib import delayed, Parallel

from bids.layout import BIDSLayout, BIDSLayoutIndexer
import numpy as np

from setup import *
from nicgiprep.utils.log_utils import LogBIDSLoader
from nicgiprep.utils.label_utils import SYNTHSEG_APARC_LUT, SYNTHSEG_APARC_DICT, SYNTHSEG_LUT, CLUSTER_DICT
from nicgiprep.utils.io_utils import create_dir, remove_dir



class Processor(object):
    """Base class for BIDS-aware neuroimaging pre-processing pipelines.

    Provides common infrastructure for loading data via PyBIDS, iterating over
    subjects and sessions, and dispatching work serially or in parallel.

    Parameters
    ----------
    bids_loader : BIDSLayout
        Initialised PyBIDS layout pointing to the root BIDS dataset and any
        required derivatives.
    subject_list : list of str, optional
        Subset of subject IDs to process. If ``None``, all subjects found in
        ``bids_loader`` are used.
    **kwargs
        Allow for other parameters when inheriting from this class. Forwarded to ``_build_processor``.

    Attributes
    ----------
    bids_loader : BIDSLayout
        Active PyBIDS layout (may be swapped per-subject during processing).
    subject_list : list of str
        Subject IDs that will be processed.
    pipeline_is_initialized : bool
        Flag set to ``True`` after ``_on_pipeline_init`` is called.
    bids_logger : LogBIDSLoader
        Helper for validating the cardinality of BIDS file queries.
    tmp_dir : str
        Path to a temporary working directory used during processing.
    labels_lut : dict
        Label lookup table mapping integer label IDs to channel indices.
    labels_dict : dict
        Mapping from integer label IDs to human-readable label names.
    """

    def __init__(self, bids_loader, subject_list=None, **kwargs):
        """
        Parameters
        ----------
        bids_loader : BIDSLayout
            Initialised PyBIDS layout.
        subject_list : list of str, optional
            Subject IDs to process. If ``None``, all subjects in the layout
            are used.
        **kwargs
            Forwarded to :meth:`_build_processor`.
        """
        self.bids_loader = bids_loader
        self.subject_list = bids_loader.get_subjects() if subject_list is None else subject_list

        self.bids_logger = LogBIDSLoader(num_files=1)
        self._build_processor()

    def _build_processor(self, **kwargs):
        """Initialise pipeline-specific state and BIDS entity filters.

        Sets common parameters between different pre-processing pipelines:
        * ``tmp_dir``: used to store temporal, intermediate results, 
        * ``seg_entities``: entities used to save/retrieve base-segmented T1w  of a given subject/session, 
        * ``bf_entities``: entities used to save/retrieve base-preprocessed T1w  of a given subject/session, 
        * ``labels_lut``: used for computing segmentation posteriors,
        * ``labels_dict``: used to save/retrieve sepecific volumes relating label number and label name. 
        
        Subclasses should call ``super()._build_processor()``
        and then extend or override these attributes.
        """
        self.tmp_dir = kwargs.get('tmp_dir', TMP_DIR)
        create_dir(self.tmp_dir)

        self.seg_entities = {'scope': 'preproc', 'extension': 'nii.gz', 'suffix': ['T1wdseg', 'dseg']}
        self.bf_entities = {'scope': 'preproc', 'extension': 'nii.gz', 'suffix': 'T1w', 'acquisition': [None, 'orig']}

        self.labels_lut = SYNTHSEG_APARC_LUT
        self.labels_dict = SYNTHSEG_APARC_DICT

        self.pipeline_is_initialized = False

    def build_path(self, entities, absolute_paths=False, validate=False, strict=True):
        """Construct a relative valid BIDS file path from an entity dictionary. It follows the convention 
        specified in this NicGi-Prep

        Filters ``entities`` to the keys recognised by the project's BIDS path
        patterns before delegating to ``BIDSLayout.build_path``.

        Parameters
        ----------
        entities : dict
            BIDS entity key/value pairs (e.g. ``{'subject': '001',
            'suffix': 'T1w', 'extension': '.nii.gz'}``).
        absolute_paths : bool, optional
            Return an absolute path. Default is ``False``.
        validate : bool, optional
            Validate the resulting path against BIDS rules. Default is
            ``False``.
        strict : bool, optional
            Raise an error if no pattern matches. Default is ``True``.

        Returns
        -------
        str or None
            The constructed path string, or ``None`` if no pattern matched
            and ``strict`` is ``False``.
        """
        entities = {k: v for k, v in entities.items() if k in filename_entities}
        return self.bids_loader.build_path(entities, absolute_paths=absolute_paths, path_patterns=BIDS_PATH_PATTERN,
                                           strict=strict, validate=validate)

    def _name(self):
        """Return the human-readable pipeline name used in console banners.

        Returns
        -------
        str
            Empty string in the base class; subclasses should override.
        """
        return 'base'

    def get_subjects(self, uslr=True):
        """Return the list of subjects in the dataset available for processing.

        Parameters
        ----------
        uslr : bool, optional
            If ``True`` (default), restrict to subjects that have been pre-processed cross-sectionally and
            available for USLR processing. It checks that at least one session has been properly segmented

        Returns
        -------
        list of str
            Subject IDs.
        """
        subjects = self.bids_loader.get_subjects()
        if uslr:
            subjects = list(filter(lambda s: len(self._get_data(**{'subject': s, **self.seg_entities},
                                                                ignore_check=True)) > 0 is not None, subjects))

        return subjects

    def _get_timepoints(self, subject, uslr=True):
        """Return the session IDs available for a given subject.

        Parameters
        ----------
        subject : str
            Subject ID.
        uslr : bool, optional
            If ``True`` (default), restrict to subjects that have been pre-processed cross-sectionally and
            available for USLR processing. It checks that at least one session has been properly segmented

        Returns
        -------
        list of str
            Session IDs as returned by ``BIDSLayout.get_session``.
        """
        timepoints = self.bids_loader.get_session(subject=subject)
        if uslr:
            timepoints = list(filter(lambda tp: self._get_data(**{
                'session': tp, 'subject': subject, **self.seg_entities}, verbose=False) is not None, timepoints))

        return timepoints

    def _get_data(self, ignore_check=False, curr_len=None, verbose=True, **kwargs):
        """Query the BIDS layout for a single file matching the given entities.

        Parameters
        ----------
        ignore_check : bool, optional
            If ``True``, return the raw file list without cardinality checks.
            Default is ``False``.
        curr_len : int or None, optional
            Expected number of matching files for the cardinality check.
            ``None`` defaults to exactly one file.
        verbose : bool, optional
            Print a warning when the query returns an unexpected number of
            files. Default is ``True``.
        **kwargs
            BIDS entity filters passed directly to ``BIDSLayout.get``.

        Returns
        -------
        BIDSFile or list or None
            A single ``BIDSFile`` when exactly one match is found, the raw list
            when ``ignore_check=True``, or ``None`` if the cardinality check
            fails.
        """
        file_list = self.bids_loader.get(**kwargs)
        if ignore_check:
            return file_list

        file_flag = self.bids_logger.check_length(file_list, curr_len=curr_len)
        if file_flag['exit_code'] == -1:
            if verbose:
                print('[warning]', end=' ', flush=True)
                print(file_flag['log'], end=' ', flush=True)
                print(' --> Entities: ' + ','.join(['<' + str(k) + ':' + str(v) + '>' for k, v in kwargs.items()]))
            raw_file = None
        else:
            raw_file = file_flag['file']

        return raw_file

    def _get_entities(self, file):
        """Extract the subset of BIDS entities relevant to filename construction.

        Parameters
        ----------
        file : BIDSFile
            A PyBIDS file object.

        Returns
        -------
        dict
            Entity key/value pairs filtered to those in ``filename_entities``.
        """
        return {k: v for k, v in file.entities.items() if k in filename_entities}

    def _on_pipeline_init(self):
        """Mark the pipeline as initialised and print a console banner.

        Sets ``pipeline_is_initialized`` to ``True`` and prints a decorated
        header with the pipeline name (if non-empty and not yet initialised).
        """
        self.pipeline_is_initialized = True
        name = self._name()
        if len(name) > 0 and self.pipeline_is_initialized is False:
            print('\n\n\n\n\n')
            print('# ' + '-'.join([''] * (len(name) + 7)) + ' #')
            print('#    ' + name + '    #')
            print('# ' + '-'.join([''] * (len(name) + 7)) + ' #')
            print('\n\n')

    def _update_subject_layout(self, subject):
        """Rebuild the BIDS layout restricted to a single subject.

        Replaces ``self.bids_loader`` with a new layout whose indexer ignores
        all subjects other than ``subject``, reducing query overhead.

        Parameters
        ----------
        subject : str
            Subject ID to keep in the layout index.
        """
        rawdir = self.bids_loader.root
        derivatives = self.bids_loader.derivatives.keys()

        indexer = BIDSLayoutIndexer(validate=False, ignore='sub-(?!' + subject + ')(.*)$', index_metadata=False)
        bids_kwargs = {'validate': False, 'indexer': indexer}

        bids_loader = BIDSLayout(root=rawdir, **bids_kwargs)
        bids_loader.add_derivatives([DIR_PIPELINES[d] for d in derivatives], **bids_kwargs)

        self.bids_loader = bids_loader

    def _update_full_layout(self):
        """Rebuild the BIDS layout without subject restriction.

        Replaces ``self.bids_loader`` with a full dataset layout after
        per-subject processing is complete.
        """
        rawdir = self.bids_loader.root
        derivatives = self.bids_loader.derivatives.keys()

        indexer = BIDSLayoutIndexer(validate=False, index_metadata=False)
        bids_kwargs = {'validate': False, 'indexer': indexer}

        bids_loader = BIDSLayout(root=rawdir, **bids_kwargs)
        bids_loader.add_derivatives([DIR_PIPELINES[d] for d in derivatives], **bids_kwargs)

        self.bids_loader = bids_loader

    def _get_subject_info(self, subject):
        """Load the sessions TSV for a subject as a DataFrame indexed by session ID.

        Parameters
        ----------
        subject : str
            Subject ID.

        Returns
        -------
        pandas.DataFrame or None
            Session-level metadata, or ``None`` if no sessions TSV is found.
        """
        sess_df = None
        sess_tsv = self._get_data(suffix='sessions', extension='tsv', subject=subject)
        if sess_tsv:
            sess_df = sess_tsv.get_df()
            sess_df = sess_df.set_index('session_id')
            sess_df = sess_df[~sess_df.index.duplicated(keep='last')]

        return sess_df

    def _get_participant_info(self):
        """Load the participants TSV as a DataFrame indexed by participant ID.

        Returns
        -------
        pandas.DataFrame or None
            Participant-level metadata, or ``None`` if no participants TSV
            is found.
        """
        part_df = None
        part_tsv = self._get_data(suffix='participants', extension='tsv')
        if part_tsv:
            part_df = part_tsv.get_df()
            part_df = part_df.set_index('participant_id')
            part_df = part_df[~part_df.index.duplicated(keep='last')]

        return part_df

    def _undo_one_hot(self, y, dtype='float32'):
        """Convert a one-hot channel index array back to integer label values.

        Parameters
        ----------
        y : np.ndarray
            Array of channel indices (output of ``np.argmax`` over the label
            axis).
        dtype : str, optional
            NumPy dtype of the output. Default is ``'float32'``.

        Returns
        -------
        np.ndarray
            Integer label map with the same shape as ``y``.
        """
        y_true = np.zeros_like(y)
        for ul, it_ul in self.labels_lut.items():
            y_true[y == it_ul] = ul

        return y_true.astype(dtype)

    def process_scan(self, subject, session, modality, force_flag=False, **kwargs):
        """Run the pipeline for a single session.

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.
        modality : str
            Modality ID.
        force_flag : bool, optional
            If ``True``, reprocess even when outputs already exist. Default
            is ``False``.
        **kwargs
            Passed to the pipeline-specific implementation.

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
        """
        raise NotImplementedError

    def process_session(self, subject, session, force_flag=False, **kwargs):
        """Run the pipeline for a single session.

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.
        force_flag : bool, optional
            If ``True``, reprocess even when outputs already exist. Default
            is ``False``.
        **kwargs
            Passed to the pipeline-specific implementation.

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
        """
        raise NotImplementedError

    def process_subject(self, subject, force_flag=False, **kwargs):
        """Run the pipeline for a single subject.

        Parameters
        ----------
        subject : str
            Subject ID.
        force_flag : bool, optional
            If ``True``, reprocess even when outputs already exist. Default
            is ``False``.
        **kwargs
            Passed to the pipeline-specific implementation.

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
        """
        raise NotImplementedError

    def process_parallel(self, num_cores, **kwargs):
        """Run ``process_subject`` for all subjects in parallel using threads.

        Parameters
        ----------
        num_cores : int
            Number of parallel worker threads.
        **kwargs
            Forwarded to ``process_subject`` for each subject.

        Returns
        -------
        list
            One entry per subject: ``None`` on success, or the subject ID
            string when processing raised an exception.
        """
        self._on_pipeline_init()

        def _run(processing, subject, **kwargs):
            """Run one subject in a thread; return the subject ID on failure."""
            processing._update_subject_layout(subject)
            try:
                processing.process_subject(subject, **kwargs)
            except:
                return subject

        results = Parallel(n_jobs=num_cores, backend='threading')(delayed(_run)(
            copy.copy(self), subject, **kwargs) for subject in self.subject_list)

        print('Subjects that failed: ')
        print('\n'.join([r for r in results if r is not None]))
        return results

    def process(self, **kwargs):
        """Run ``process_subject`` sequentially for all subjects.

        Subjects that raise an exception are collected and printed at the
        end rather than aborting the run.

        Parameters
        ----------
        **kwargs
            Forwarded to ``process_subject`` for each subject.
        """
        self._on_pipeline_init()

        subjects_failed = []
        for subject in self.subject_list:
            self._update_subject_layout(subject)
            try:
                retcode = self.process_subject(subject, **kwargs)
                if retcode is None or retcode['exit'] != 1:
                    subjects_failed.append(subject)
            except:
                subjects_failed += [subject]

        self._update_full_layout()
        print('Subjects that failed: ')
        print('\n'.join(subjects_failed))


