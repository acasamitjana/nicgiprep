"""
Processing pipeline classes for multimodal (multi-contrast) MRI data in BIDS format.

This module implements the joint, unbiased multimodal registration. It reuses the same rigid
spanning-tree estimator as the longitudinal processing pipeline
(:mod:`nicgiprep.pipelines.longitudinal`), the only conceptual change being that
the graph vertices are the **image modalities of a single session** instead of
the **timepoints of a single subject**.

"""

import copy
import itertools
import pdb
import traceback
import tqdm
from os import makedirs
from os.path import join, dirname, basename, exists
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib
import pandas as pd
from bids.layout import BIDSFile
from scipy.ndimage import gaussian_filter
from skimage.morphology import ball, binary_dilation
from sympy.vector import Cross

from nicgiprep.pipelines import CrossSectionalProcessor
from setup import *
from nicgiprep.pipelines.base import Processor
from nicgiprep.pipelines.cross_sectional import T1wSegmentationProcessor, T1wBiasCorrectionProcessor
from nicgiprep.pipelines.longitudinal import USLRLinear
from nicgiprep.utils.io_utils import create_dir, save_volume, remove_dir, ProcessResult
from nicgiprep.utils.def_utils import (
    vol_resample_fast,
    network_space,
    create_empty_template,
    getM,
)
from nicgiprep.utils.fn_utils import (
    rescale_voxel_size,
    compute_centroids_ras,
    one_hot_encoding,
)
from nicgiprep.utils.preprocessing_utils import bias_field_corr
from nicgiprep.utils.label_utils import SYNTHSEG_LUT, SYNTHSEG_GMM_ONTOLOGY, labels_registration


class MMProcessor(CrossSectionalProcessor):
    """Base class for multi-modal processing pipelines.

    Extends :class:`~nicgiprep.pipelines.base.Processor` with BIDS entity
    templates for the multimodal session-space outputs (per-modality affines,
    images, masks and segmentations in the session common space, the session
    template, and the network-space ``v2r`` array).

    Attributes
    ----------
    modalities : list of str
        MRI modality suffixes considered by the pipeline. Limited to anatomical
        MRI contrasts (no CT/PET/fMRI). Defaults to ``['T1w', 'T2w', 'FLAIR',
        'PDw']``.
    template_modality : str
        Modality used as the session-specific template ``S0``. Defaults to
        ``'T1w'``.
    net_shape : tuple of int
        Spatial shape of the 1 mm isotropic session/common space.
    """

    #: Anatomical reference space where all the outputs are defined
    SPACE = 'session'

    #: Anatomical MRI contrasts handled by the multimodal pipeline.
    DEFAULT_MODALITIES = ["T1w", "T2w", "FLAIR", "PDw"]

    #: Modality elected as the session template.
    TEMPLATE_MODALITY = "T1w"


    def _build_processor(self, **kwargs) -> None:
        """Extend the base processor with multimodal entity templates.

        On top of the attributes set by
        :meth:`Processor._build_processor`, this adds:

        - ``modalities`` / ``template_modality`` — modality configuration.
        - ``net_shape`` — common-space spatial shape.
        - ``aff_graph_entities`` — entities for the per-modality
          modality-to-template affine files.
        - ``im_graph_entities`` / ``mask_graph_entities`` — entities for the
          per-modality images and masks resampled into the session space.
        - ``template_entities`` — entities for the session template.
        - ``net_v2r_entities`` — entities for the session-space ``v2r`` array.

        Parameters
        ----------
        **kwargs
            Optional ``modalities`` (list of str) and ``template_modality``
            (str) overrides, plus any keyword arguments forwarded to
            :meth:`Processor._build_processor`.

        Returns
        -------
        None
        """
        super()._build_processor(**kwargs)

        self.modalities = kwargs.get("modalities", list(self.DEFAULT_MODALITIES))
        self.template_modality = kwargs.get("template_modality", self.TEMPLATE_MODALITY)

        self.net_shape = (192, 192, 192)

        # Per-modality affine mapping the raw modality to the session template.
        # The modality is encoded in the ``desc`` entity (e.g. ``desc-T1wtosession``).
        self.im_session_entities = {
            "space": self.SPACE,
            "extension": ".nii.gz",
        }
        self.aff_graph_entities = {
            "space": self.SPACE,
            "extension": ".npy",
        }

        # Per-modality image / mask / seg resampled into the session space.
        # self.im_graph_entities = {"space": self.SPACE, "extension": ".nii.gz"}
        # self.mask_graph_entities = {"space": self.SPACE, "extension": ".nii.gz"}

        # Session template (S0): the aligned template-modality image + derivatives.
        self.template_entities = {
            "space": self.SPACE,
            "datatype": "utils",
            "acquisition": "1",
            "suffix": "empty",
            "desc": "template",
            "extension": ".nii.gz",
        }


        #: Derivatives directory key for the cross-sectional preprocessing outputs.
        self.pipeline_cross_dir = "nicgiprep-cross" if 'pipeline-cross' not in kwargs else kwargs['pipeline-cross']

        #: Derivatives directory key / pybids scope for the multimodal outputs.
        self.pipeline_dir = "nicgiprep-mm"

    def _name(self) -> str:
        """Return the display name of this pipeline."""
        return "MMProcessor"

    # ------------------------------------------------------------------ #
    #  Per-modality BIDS entity helpers                                   #
    # ------------------------------------------------------------------ #
    def _seg_suffix(self, modality: str) -> str:
        """Return the SynthSeg segmentation suffix for a modality.

        Parameters
        ----------
        modality : str
            Modality suffix (e.g. ``'T1w'``).

        Returns
        -------
        str
            The SynthSeg segmentation suffix, e.g. ``'T1wsynthseg'``.
            Note: ``dseg`` is reserved for SuperSynth segmentations only.
        """
        return modality + "synthseg"

    def _mask_suffix(self, modality: str) -> str:
        """Return the brain-mask suffix for a modality.

        Parameters
        ----------
        modality : str
            Modality suffix (e.g. ``'T1w'``).

        Returns
        -------
        str
            The brain-mask suffix, e.g. ``'T1wmask'``.
        """
        return modality + "mask"

    def _modality_datatype(self, modality: str) -> str:
        """Return the BIDS datatype directory for a cross-sectional modality output.

        The template modality (T1w) is written to ``anat/``; all other
        modalities go to ``utils/`` as intermediate artefacts.

        Parameters
        ----------
        modality : str
            Modality suffix (e.g. ``'T2w'``).

        Returns
        -------
        str
            ``'anat'`` for the template modality, ``'utils'`` otherwise.
        """
        return "anat" if modality == self.template_modality else "utils"

    def _modality_image_entities(self, modality: str) -> Dict:
        """Build BIDS entities selecting a modality's cross-sectionally preprocessed image.

        Reads from the ``nicgiprep-cross`` derivatives directory, routing the
        template modality (T1w) to ``anat/`` and all others to ``utils/``.

        Parameters
        ----------
        modality : str
            Modality suffix (e.g. ``'T2w'``).

        Returns
        -------
        dict
            BIDS entity filters for the preprocessed modality image.
        """
        entities = {
            "scope": self.pipeline_cross_dir,
            "extension": ".nii.gz",
            "suffix": modality,
            "acquisition": [None, "orig"],
        }
        if modality == self.template_modality:
            entities["datatype"] = "anat"
        return entities

    def _modality_seg_entities(self, modality: str) -> Dict:
        """Build BIDS entities selecting a modality's SynthSeg segmentation.

        All SynthSeg segmentations (including T1w, produced by the
        cross-sectional pipeline) live in ``utils/`` under ``nicgiprep-cross``.

        Parameters
        ----------
        modality : str
            Modality suffix (e.g. ``'T2w'``).

        Returns
        -------
        dict
            BIDS entity filters for the modality's SynthSeg segmentation.
        """
        return {
            "scope": self.pipeline_cross_dir,
            "extension": ".nii.gz",
            "suffix": self._seg_suffix(modality),
        }

    # def _aff_desc(self, modality: str) -> str:
    #     """Build the ``desc`` value encoding a modality-to-template affine.
    #
    #     Parameters
    #     ----------
    #     modality : str
    #         Modality suffix (e.g. ``'T1w'``).
    #
    #     Returns
    #     -------
    #     str
    #         The ``desc`` entity value, e.g. ``'T1wtosession'``.
    #     """
    #     return modality + "to" + self.SPACE

    # ------------------------------------------------------------------ #
    #  Discovery helpers                                                  #
    # ------------------------------------------------------------------ #
    def _get_modalities(self, subject: str, session: str) -> List[str]:
        """List the configured modalities available *and* segmented for a session.

        A modality is considered usable only when both its preprocessed image
        and a segmentation (needed for the parcellation-based rigid
        registration) are present.

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.

        Returns
        -------
        list of str
            Modality suffixes available for joint registration, ordered as in
            :attr:`modalities`.
        """
        available = []
        for modality in self.modalities:
            im_file = self._get_data(
                **{
                    "subject": subject,
                    "session": session,
                    **self._modality_image_entities(modality),
                },
                verbose=False
            )
            seg_file = self._get_data(
                **{
                    "subject": subject,
                    "session": session,
                    **self._modality_seg_entities(modality),
                },
                verbose=False
            )
            if im_file is not None and seg_file is not None:
                available.append(modality)

        return available


class MultiModalSegmentationProcessor(MMProcessor, T1wSegmentationProcessor):
    """Run SynthSeg on *every* MRI modality of each session.

    The base :class:`~nicgiprep.pipelines.cross_sectional.SynthSegProcessor`
    only segments the T1w image. The multimodal registration step, however,
    needs a parcellation per modality (SynthSeg is contrast-agnostic), so this
    subclass selects one image per modality/session and produces a
    ``<mod>dseg`` segmentation for each.

    It reuses the base :meth:`SynthSegProcessor.process` machinery unchanged:
    that method batches all collected input/output paths through a single
    ``mri_synthseg --robust --parc`` call. Only the per-subject path collection
    is generalised here.

    Notes
    -----
    The base ``process`` aggregates per-session volume tables by scanning for
    ``T1wdseg`` volume files only, so the population ``*_vols.csv`` remains
    T1w-based; the non-T1w segmentations are still written to disk and consumed
    by :class:`MultiMRIProcessor`.
    """

    def _name(self) -> str:
        """Return the display name of this pipeline."""
        return "MultiModalSynthSegSegmentation"

    def _build_processor(self, **kwargs) -> None:
        """Initialise base state and the list of modalities to segment.

        Parameters
        ----------
        **kwargs
            Optional ``modalities`` (list of str) override, plus any keyword
            arguments forwarded to :meth:`SynthSegProcessor._build_processor`.

        Returns
        -------
        None
        """
        super()._build_processor(**kwargs)
        self.modalities = kwargs.get("modalities", list(self.DEFAULT_MODALITIES))


    def _write_session_inputs_log(
        self, subject: str, session: str, selected: List[Tuple[str, str]]
    ) -> None:
        """Write (or overwrite) a TSV recording which raw file was selected per modality.

        The file is written to
        ``nicgiprep-cross/sub-<subject>/ses-<session>/sub-<subject>_ses-<session>_mm-inputs.tsv``
        and contains columns ``modality`` and ``raw_path``.

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.
        selected : list of (modality, path) tuples
            One entry per modality that was selected for this session.
        """
        out_dir = join(
            DIR_PIPELINES[self.pipeline_cross_dir],
            "sub-" + subject,
            "ses-" + session,
        )
        makedirs(out_dir, exist_ok=True)
        tsv_path = join(
            out_dir, "sub-" + subject + "_ses-" + session + "_mm-inputs.tsv"
        )
        df = pd.DataFrame(selected, columns=["modality", "raw_path"])
        df.to_csv(tsv_path, sep="\t", index=False)

    def process_subject(
        self,
        subject: str,
        force_flag: bool = False,
        session_list: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[List, List, List, Dict, List]:
        """Collect SynthSeg I/O paths for non-T1w modalities of every session.

        The T1w image must already be processed by the cross-sectional pipeline
        before this method is called. Specifically, the cross-sectional
        ``anat/`` folder must contain a file ending with ``T1w.nii.gz`` (the
        bias-corrected image) and a file ending with ``T1wdseg.nii.gz`` (the
        SuperSynth segmentation). If these are absent for a session, the session
        is skipped with a warning.

        Only non-T1w modalities are queued — T1w SynthSeg segmentation is
        produced by the cross-sectional pipeline and lives in ``utils/`` as
        ``T1wsynthseg.nii.gz``. Non-T1w segmentations are written as
        ``{modality}synthseg.nii.gz`` in ``utils/``.

        For each session processed, writes a TSV log recording which raw image
        was selected for each modality (see :meth:`_write_session_inputs_log`).

        Parameters
        ----------
        subject : str
            Subject ID.
        force_flag : bool, optional
            If ``True``, re-queue modalities even when a segmentation already
            exists. Default is ``False``.
        session_list : list of str, optional
            Restrict processing to these session IDs. If ``None``, all
            sessions for the subject are processed.
        **kwargs
            Ignored; present for API compatibility.


        Returns
        -------
        tuple of list/doct
            ``(input_files, res_files, output_files, vol_files,
            )`` — one entry per session queued.
        """
        input_files, res_files, output_files, found_files, supersynth_files = (
            [],
            [],
            [],
            [],
            {'subject': [], 'session': [], 'input_file': [], 'output_fname': [], 'mode': [], 'tmp_dir': [], 'run': []},
        )

        sessions = self.bids_loader.get_session(subject=subject)
        if session_list is not None:
            sessions = [s for s in sessions if s in session_list]

        for sess_id in sessions:
            tmp_dir = join(
                TMP_DIR,
                "supersynth_process",
                "sub-" + subject,
                "ses-" + sess_id)

            cross_anat_dir = join(
                DIR_PIPELINES[self.pipeline_cross_dir],
                "sub-" + subject,
                "ses-" + sess_id,
                "anat",
            )
            cross_utils_dir = join(
                DIR_PIPELINES[self.pipeline_cross_dir],
                "sub-" + subject,
                "ses-" + sess_id,
                "utils",
            )

            mm_anat_dir = join(
                DIR_PIPELINES[self.pipeline_dir],
                "sub-" + subject,
                "ses-" + sess_id,
                "anat",
            )

            mm_utils_dir = join(
                DIR_PIPELINES[self.pipeline_dir],
                "sub-" + subject,
                "ses-" + sess_id,
                "utils",
            )

            # Collect raw modalities available for this session (T1w included for logging).
            session_log: List[Tuple[str, str]] = []

            image_files = self._select_images(subject, sess_id, **{'modality': ['T1w', 'T2w', 'PD', 'FLAIR']})
            if image_files is None:
                continue

            if isinstance(image_files, BIDSFile):
                image_files = [image_files]

            found_files.extend(image_files)

            for image_file in image_files:
                image_entities = dict(image_file.entities)
                image_entities["acquisition"] = "1"

                synthseg_entities = copy.deepcopy(image_entities)
                synthseg_entities['suffix'] += "synthseg"

                anat_res = basename(self.build_path(image_entities))
                anat_synthseg = basename(self.build_path(synthseg_entities))
                anat_seg = image_file.filename.replace(".nii.gz", "dseg.nii.gz")

                session_log.append((image_entities['suffix'], image_file.path))

                # Synthseg
                if not exists(join(mm_utils_dir, anat_synthseg)) or force_flag:
                    if not exists(mm_utils_dir): os.makedirs(mm_utils_dir)
                    if exists(join(cross_utils_dir, anat_synthseg)):
                        subprocess.call(
                            ["cp", join(cross_utils_dir, anat_synthseg), join(mm_utils_dir, anat_synthseg)]
                        )

                    else:
                        proxy = nib.load(image_file.path)
                        run_code = self._check_file(proxy)

                        if run_code["run_flag"]:
                            input_files += [image_file.path]
                            res_files += [join(mm_utils_dir, anat_res)]
                            output_files += [join(mm_utils_dir, anat_synthseg)]

                        else:
                            with open(join(mm_utils_dir, "excluded_file.txt"), "w") as f:
                                f.write(run_code["exit_message"])

                # SuperSynth
                if not exists(join(mm_anat_dir, anat_seg)):
                    if not exists(mm_anat_dir): os.makedirs(mm_anat_dir)
                    if (exists(join(cross_anat_dir, anat_seg)) and image_entities['suffix'] == 'T1w') and not force_flag:
                        subprocess.call(
                            ["ln", '-s', join(cross_anat_dir, anat_seg), join(mm_anat_dir, anat_seg)]
                        )
                    else:
                        os.makedirs(join(tmp_dir, image_file.filename), exist_ok=True)
                        supersynth_files['subject'] += [subject]
                        supersynth_files['session'] += [sess_id]
                        supersynth_files['input_file'] += [f"{image_file.path}"]
                        supersynth_files['tmp_dir'] += [join(tmp_dir, image_file.filename)]
                        supersynth_files['mode'] += ["invivo"]
                        supersynth_files['output_fname'] += [join(mm_anat_dir, anat_seg)]
                        if not exists(join(tmp_dir, image_file.filename, "segmentation.mgz")) or force_flag:
                            supersynth_files['run'] += [True]
                        else:
                            supersynth_files['run'] += [False]

            self._write_session_inputs_log(subject, sess_id, session_log)

        return input_files, res_files, output_files, supersynth_files, found_files


    def _seg_suffix(self, modality: str) -> str:
        """Return the SynthSeg output suffix for a modality.

        Parameters
        ----------
        modality : str
            Modality suffix (e.g. ``'T2w'``).

        Returns
        -------
        str
            The SynthSeg segmentation suffix, e.g. ``'T2wsynthseg'``.
        """
        return modality + "synthseg"


class MultiModalBiasCorrectionProcessor(MMProcessor, T1wBiasCorrectionProcessor):
    """Bias-field correction and intensity normalisation for all MRI modalities.

    Runs the same EM-based bias-field correction and WM-mean normalisation as
    :class:`~nicgiprep.pipelines.cross_sectional.BiasCorrectionProcessor`, but
    applied to every modality rather than T1w alone.

    Reads SynthSeg segmentations and 1 mm resampled images from
    ``nicgiprep-cross`` (written by :class:`MultiModalSynthSegProcessor`), and
    writes the bias-corrected images back to the same directories:

    - T1w → ``nicgiprep-cross/sub-<s>/ses-<s>/anat/``
    - All other modalities → ``nicgiprep-cross/sub-<s>/ses-<s>/utils/``

    The corrected image is saved under the raw filename (no ``acq`` entity) so
    that :class:`MultiMRIProcessor` can find it via
    :meth:`MMProcessor._modality_image_entities` (which filters for
    ``acquisition=[None, 'orig']``).
    """

    def _name(self) -> str:
        """Return the display name of this pipeline."""
        return "MultiModalBiasFieldCorrection"


    def process_subject(self,
                        subject: str,
                        session_list: Optional[List[str]] = None,
                        force_flag: bool=False, **kwargs):
        """Run bias-field correction for all sessions of one subject.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str, optional
            Restrict processing to these session IDs. If ``None``, all
            sessions for the subject are processed.
        force_flag : bool, optional
            If ``True``, reprocess sessions even when outputs already exist.
            Default is ``False``.
        **kwargs
            Ignored; present for API compatibility.

        Returns
        -------
        dict
            Result dict with keys:
            - ``exit_code`` (int): ``0`` on success / already processed, ``-1`` on failure.
            - ``message`` (str): human-readable description of the outcome.
            - ``images_processed`` (int): number of images successfully processed for this subject.
            - ``images_failed`` (int): number of images that failed processing for this subject. This may have
                                       numerous causes, among them, wrong segmentation or poor quality
        """
        print("\nSubject: " + subject, end='\n')

        sessions = self._get_sessions(subject=subject)
        if session_list is not None:
            sessions = [s for s in sessions if s in session_list]

        exit_dict = {"images_processed": [], "images_failed": [], "exit_code": 0, "message": ""}
        for sess_id in tqdm.tqdm(sessions, leave=False, desc="Processing sessions"):
            # input segs
            synthseg_entities = copy.copy(self.seg_entities)
            synthseg_entities["scope"] = [self.pipeline_dir]
            synthseg_entities["suffix"] = ["T1wsynthseg", "T2wsynthseg", "FLAIRsynthseg", 'PDsynthseg']
            synthseg_entities["space"] = [None]
            synthseg_entities["desc"] = [None]
            synthseg_entities["datatype"] = ["utils"]
            seg_files = self._get_data(
                **{"session": sess_id, "subject": subject, **synthseg_entities}, ignore_check=True
            )
            if seg_files is None:
                continue

            cross_utils = join(
                DIR_PIPELINES[self.pipeline_cross_dir],
                "sub-" + subject,
                "ses-" + sess_id,
                "utils",
            )

            mm_anat = join(
                DIR_PIPELINES[self.pipeline_dir],
                "sub-" + subject,
                "ses-" + sess_id,
                "anat",
            )

            for seg_file in tqdm.tqdm(seg_files, leave=False, desc="Processing images     "):
                raw_entities = self._get_entities(seg_file)
                raw_entities["extension"] = ".nii.gz"
                raw_entities["suffix"] = raw_entities["suffix"].split('synthseg')[0]
                raw_entities["scope"] = "raw"
                raw_entities["acquisition"] = [None, "orig"]
                raw_entities.pop("datatype", None)
                if 'run' not in raw_entities: raw_entities['run'] = None

                raw_file = self._get_data(**raw_entities)
                if raw_file is None:
                    continue

                ret_code = {'exit_code': -1}
                if isinstance(raw_file, BIDSFile):
                    output_filepath = join(mm_anat, raw_file.filename)
                    if exists(output_filepath):
                        ret_code = {'exit_code': 0}

                    elif exists(join(cross_utils, raw_file.filename)):
                        subprocess.call(
                            ["ln", "-s", join(cross_utils, raw_file.filename), output_filepath]
                        )

                    else:
                        ret_code = self.process_image(im_file=raw_file,
                                                      seg_file=seg_file,
                                                      output_filepath=output_filepath,
                                                      force_flag=force_flag)

                if ret_code['exit_code'] == 0:
                    exit_dict['images_processed'] += [seg_file.path]
                else:
                    exit_dict['images_failed'] += [seg_file.path]
                    exit_dict['exit_code'] = 1

        exit_dict['total_images'] = len(exit_dict['images_processed']) + len(exit_dict['images_failed'])
        exit_dict['message'] = (str(len(exit_dict['images_processed'])) + "/" + str(exit_dict['total_images']) +
                                " of images were successfully processed.")
        print(' o ' + exit_dict['message'])
        print("\n")
        return exit_dict



class MultiMRIProcessor(MMProcessor, USLRLinear):
    """Joint, unbiased rigid registration of multiple MRI contrasts.

    For every session, estimates one rigid transform per modality that maps it
    into a session-specific common space lying at the centre of all modalities,
    by solving the same log-space spanning-tree problem as
    :class:`~nicgiprep.pipelines.longitudinal.USLRLinear`. The aligned
    template modality (T1w by default) is stored as the session template.

    The processing for a single session follows USLR's linear step closely:

    1. Compute RAS centroids of each modality's segmentation.
    2. Centre each modality on its centre-of-gravity (COG).
    3. Estimate pairwise rigid transforms between every pair of modalities via
       centroid SVD.
    4. Solve the spanning tree in the Lie-algebra domain to obtain one latent
       transform per modality (template -> modality).
    5. Build the unbiased session space, resample every modality into it and
       elect the aligned template modality as the session template ``S0``.
    """

    def _name(self) -> str:
        """Return the display name of this pipeline."""
        return "MultimodalRegistration"

    def _build_processor(self, **kwargs) -> None:
        """Extend the base processor with the session-space temp dir and output pipeline.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :meth:`MMProcessor._build_processor`.

        Returns
        -------
        None
        """
        super()._build_processor(**kwargs)
        self.tmp_dir = join(self.tmp_dir, "MMSessionSpace")
        create_dir(self.tmp_dir)

    def _get_process_image_file(self, subject: str, sessions: List[str]) -> pd.DataFrame | ProcessResult:
        """Create a table with all sessions and needed information to process. It contains the following columns:
            * subject
            * session_id
            * orig_image: the selected MRI modalities for the multimodal pipeline.
                          If there are multiple runs available in the rawdata, select ALL.
            * orig_seg: the supersynth segmentation of the selected T1w image for the longitudinal pipeline.
            * orig_synthseg: the synthseg segmentation of the selected T1w image for the longitudinal pipeline. It
                             will be used for rigid registration.

        Parameters
        ----------
        subject : str
            Subject ID
        sessions : List[str]
            List of session IDs to run.

        Returns
        -------
        pd.DataFrame | Process results
            dataframe containing all session to process. It define at least three columns:
            'session_id', 'orig_mri' and 'orig_seg'

        """
        sess_fpath = join(
            DIR_PIPELINES[self.pipeline_dir],
            "sub-" + subject,
            "sub-" + subject + "_sessions.tsv",
        )

        if exists(
            join(
                DIR_PIPELINES[self.pipeline_dir],
                "sub-" + subject,
                "sub-" + subject + "_sessions.tsv",
            )
        ):
            sess_df = pd.read_csv(sess_fpath, sep="\t")
            if all([s in sess_df.session_id.to_list() for s in sessions]):
                return sess_df

        else:
            os.makedirs(join(
                DIR_PIPELINES[self.pipeline_dir],
                "sub-" + subject
            ), exist_ok=True)

            sess_df = pd.DataFrame({
                "subject": [],
                "session_id": [],
                "orig_mri": [],
                "orig_seg": [],
                "orig_synthseg": [],
                "index_image": []
            })

        im_raw_ent = {"scope": "nicgiprep-mm", "extension": ".nii.gz", "suffix": ["T1w", "T2w", "PD", "FLAIR"],
                      'space': None,  'datatype': 'anat'}

        for sess_id in sessions:
            im_raw_files = self.bids_loader.get(
                **{"subject": subject, "session": sess_id, **im_raw_ent}
            )

            if len(im_raw_files) == 0:
                continue

            elif len(im_raw_files) == 1:
                continue

            row = {}
            for idx_i, im_file in enumerate(im_raw_files):
                im_ent = im_file.get_entities()
                seg_ent = im_file.get_entities()
                seg_ent["suffix"] = [im_ent["suffix"] + "dseg", im_ent["suffix"] + "dseg"]
                seg_ent["scope"] =  "nicgiprep-mm"
                if 'run' not in seg_ent.keys():
                    seg_ent["run"] = None # avoid reading the all runs if run is not in the filename
                seg_files = self._get_data(**seg_ent)

                seg_ent.pop('datatype')
                seg_ent["suffix"] = im_ent["suffix"] + "synthseg"
                seg_ent["desc"] = None # avoid reading the template image
                seg_files_synthseg = self._get_data(**seg_ent)

                row["subject"] = [subject]
                row["session_id"] = [sess_id]
                row["index_image"] = ["M" + str(idx_i)]
                row["orig_mri"] = [im_file.path]
                if seg_files is None:
                    row["orig_seg"] = [None]
                else:
                    row["orig_seg"] = [seg_files.path]


                if seg_files_synthseg is None:
                    row["orig_synthseg"] = [None]
                else:
                    row["orig_synthseg"] = [seg_files_synthseg.path]

                sess_df = pd.concat([sess_df, pd.DataFrame(row)], axis=0, ignore_index=True)

        sess_df.drop_duplicates(inplace=True)
        sess_df.to_csv(sess_fpath, sep="\t", index=False)

        return sess_df

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                      #
    # ------------------------------------------------------------------ #
    def _check_running_session(
        self, sess_df: pd.DataFrame, force_flag: bool
    ) -> ProcessResult:
        """Determine the processing checkpoint for a single session.

        Parameters
        ----------
        sess_df: pd.DataFrame
            Table with index_image as index and (subject, session_id, orig_mri, orig_synthseg) as columns.
        force_flag : bool
            If ``True``, ignore existing outputs and rerun.

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}``. Exit codes:
            ``-1`` error, ``0`` run full pipeline, ``1`` skip (already done),
            ``5`` single modality (nothing to register, link as template).
        """
        default_result = ProcessResult(exit_code=-1, message="[error] Some error during the process occurred. \n")

        try:
            # Nothing to register if there are no modalities.
            if len(sess_df) == 0:
                return ProcessResult(
                    exit_code=1,
                    message="[done] no usable image modalities found. Skipping.\n",
                )

            # A single modality cannot define a joint space: skip multimodal registration.
            if len(sess_df) == 1:
                return ProcessResult(
                    exit_code=1,
                    message="[skip] only 1 modality available. Multimodal registration requires at least 2 modalities.\n",
                )

            if any([f is None for f in sess_df['orig_synthseg']]):
                return ProcessResult(
                    exit_code=-1,
                    message="[error] missing SynthSeg segmentations in the subject/session/utils file.\n",
                )
            # Already processed: a per-modality affine exists for every modality and
            # the session template is present.
            subject = sess_df.iloc[0]["subject"]
            session = sess_df.iloc[0]["session_id"]
            # step 1: are all affine matrices computed?
            all_affs = all(
                self._get_data(
                    **{"subject": subject,
                       "session": session,
                       'suffix': self._get_entities(file)['suffix'].split('synthseg')[0] + 'aff',
                       'run': self._get_entities(file).get('run', None),
                       **self.aff_graph_entities},
                    verbose=False,
                )
                is not None
                for file in sess_df['orig_synthseg']
            )

            # step 2: is the t1w segmentation in session space available?
            sss_kwargs = self.template_entities.copy()
            sss_kwargs["subject"] = subject
            sss_kwargs["session"] = session
            sss_file = self._get_data(**sss_kwargs, verbose=False)

            # step 3: are all images resampled to session space?
            all_im_resampled = all(
                self._get_data(
                    **{"subject": subject,
                       "session": session,
                       'suffix': self._get_entities(file)['suffix'],
                       'run': self._get_entities(file).get('run', None),
                       **self.im_session_entities},
                    verbose=False,
                )
                is not None
                for file in sess_df['orig_mri']
            )

            # step 4: is the affine matrix to MNI computed?
            mni_aff_entities = {
                "subject": subject,
                "session": session,
                "space": "MNI",
                "suffix": "aff",
                "extension": ".npy",
                "scope": self.pipeline_dir
            }
            aff_MNI = self._get_data(**mni_aff_entities, verbose=False)

            if not all_affs or force_flag:
                return ProcessResult(
                    exit_code=1,
                    message="[running] subject needs to be processed."
                )

            elif sss_file is None:
                return ProcessResult(
                    exit_code=2,
                    message="[partly done] graph is solved; need to create session space."
                )

            elif not all_im_resampled:
                return ProcessResult(
                    exit_code=3,
                    message="[partly done] session space is computed; need to resample images."
                )

            elif aff_MNI is None:
                return ProcessResult(
                    exit_code=4,
                    message="[partly done] missing MNI registration from the session template."
                )

            else:
                return ProcessResult(
                    exit_code=0,
                    message="[done] session already processed. "
                    "Check the results in [..]/" + self.pipeline_dir
                    + "/sub-" + subject
                    + "/ses-" + str(session)
                    + ".\n",
                )

        except Exception as e:
            return default_result

    def _solve_graph(
            self, subject: str, sess_df: pd.DataFrame, def_dir: str, t_cog_d: dict, **kwargs
    ) -> ProcessResult:
        """Solve the rigid spanning-tree problem and save per-modality affines.

        Reuses the USLR log-space solver verbatim
        (:meth:`USLRLinear.init_st2_lineal` and
        :meth:`USLRLinear.st2_lineal_pytorch`), since the multimodal and USLR
        rigid models are mathematically identical — only the graph vertices differ
        (modalities here vs. timepoints in USLR).

        Parameters
        ----------
        subject : str
            Subject ID.
        sess_df: pd.DataFrame
            Table with index_image as index and (subject, session_id, orig_mri, orig_synthseg) as columns.
        def_dir : str
            Directory containing pairwise ``<ref>_to_<flo>.npy`` files.
        **kwargs
            Forwarded to :meth:`USLR_Linear.st2_lineal_pytorch`
            (e.g. ``n_epochs``, ``cost``, ``lr``).

        Returns
        -------
        ProcessResults
            ``{'exit_code': int, 'message': str, 'data': dict}``.
            Exit codes:
                ``-1`` error,
                ``0`` process is already completed (or has a single or no timpeoints available)
        """
        Tres = super()._solve_graph(subject, sess_df, def_dir, t_cog_d, **kwargs)
        if np.sum(np.isnan(Tres)) > 0:
            if np.sum(np.isnan(Tres)) == np.sum(np.isnan(Tres[:3, 3])):
                # nan values in the displacement, meaning that there is no translation among images
                # (probably taken at the same moment)
                Tres[:3, 3] = 0
            else:
                return {
                    "exit_code": -1,
                    "message": "[error] Something went wrong in the rigid registration step.\n",
                }

        for it_mod, (idx_row, row) in enumerate(sess_df.iterrows()):
            extra_kwargs = {
                "subject": row['subject'],
                "session": row["session_id"],
                "run": self._get_entities(row['orig_synthseg']).get('run', None),
                "suffix": self._get_entities(row['orig_synthseg'])['suffix'].split("synthseg")[0] + 'aff',
            }
            filename = self.build_path({**extra_kwargs, **self.aff_graph_entities})

            affine_matrix = Tres[..., it_mod]
            T_cog = t_cog_d[idx_row]

            output_filepath = join(DIR_PIPELINES[self.pipeline_dir], filename)
            create_dir(dirname(output_filepath))
            np.save(output_filepath, np.linalg.inv(T_cog) @ affine_matrix)

        return {
            "exit_code": 2,
            "message": "[partly done] graph computed; building session space.\n",
        }

    # ------------------------------------------------------------------ #
    #  Session space construction                                         #
    # ------------------------------------------------------------------ #
    def _create_session_space(
        self, sess_df: pd.DataFrame
    ) -> ProcessResult | None:
        """Build the unbiased session space and resample every modality into it.

        Computes an average bounding box from all modalities' brain masks
        (placed in the common frame via their solved affines), defines a 1 mm
        isotropic LIA network space, and resamples each modality's image and
        segmentation there. The aligned template modality (T1w) is additionally
        stored as the session template ``S0``.

        Arrays are kept channels-last; the label maps handled here are
        ``(X, Y, Z)`` with no channel axis.

        Parameters
        ----------
        sess_df: pd.DataFrame
            Table with index_image as index and (subject, session_id, orig_mri, orig_synthseg) as columns.

        Returns
        -------
        ProcessResult
            ``{'exit_code': int, 'message': str, 'data': dict}``.
            Exit codes:
                ``-1`` error,
                ``0`` process is already completed (or has a single or no timpeoints available)
        """

        # Gather solved affines and dilated masks in the common frame.
        masks_dilated = []
        t1w_onehot = []
        for idx_row, row in sess_df.iterrows():
            synthseg_file = row['orig_synthseg']
            aff_file = self._get_data(
                **{"subject": row["subject"],
                   "session": row["session_id"],
                   'suffix': self._get_entities(synthseg_file)['suffix'].split('synthseg')[0] + 'aff',
                   'run': self._get_entities(synthseg_file).get('run', None),
                   **self.aff_graph_entities},
                verbose=False,
            )

            if aff_file is None or synthseg_file is None:
                return ProcessResult(
                    exit_code=-1,
                    message="[error] Something went wrong in the rigid registration step.\n",
                )

            aff = np.load(aff_file)
            if np.sum(np.isnan(aff)) > 0:
                return ProcessResult(
                    exit_code=-1,
                    message="[error] Something went wrong in the rigid registration step.\n",
                )

            # aff_dict[idx_row] = aff

            seg_proxy = nib.load(synthseg_file)
            seg_arr = np.array(seg_proxy.dataobj)
            if self._get_entities(row['orig_mri'])['suffix'] == 'T1w':
                onehot_arr = one_hot_encoding(seg_arr, categories=self.synthseg_lut )
                t1w_onehot.append(
                    nib.Nifti1Image(onehot_arr.astype('float32'), np.linalg.inv(aff) @ seg_proxy.affine)
                )

            mask_arr = (seg_arr > 0) & (seg_arr != 24)

            mask_dilated_arr = binary_dilation(mask_arr, ball(3)).astype("float")
            masks_dilated.append(
                nib.Nifti1Image(mask_dilated_arr, np.linalg.inv(aff) @ seg_proxy.affine)
            )



        # Define the common (network) space centred on the union bounding box.
        _, template_v2r, template_size = create_empty_template(
            masks_dilated, margin_bb=5
        )

        sss_kwargs = self.template_entities.copy()
        sss_kwargs["subject"] = sess_df["subject"].iloc[0]
        sss_kwargs["session"] = sess_df["session_id"].iloc[0]

        root_dir = DIR_PIPELINES[self.pipeline_dir]
        sss_filepath = join(root_dir, self.build_path(sss_kwargs))

        sss_kwargs["suffix"] = "T1wsynthseg"
        sss_seg_filepath = join(root_dir, self.build_path(sss_kwargs))

        save_volume(
            np.zeros(template_size),
            template_v2r,
            sss_filepath,
        )
        if len(t1w_onehot) > 0:
            proxytemp = nib.Nifti1Image(np.zeros(template_size), template_v2r)
            arr_template_onehot = np.zeros(template_size + (len(self.synthseg_lut),), dtype='float32')
            for proxyseg in t1w_onehot:
                arr_template_onehot += vol_resample_fast(proxytemp, proxyseg, return_np=True)

            arr_template_seg = np.argmax(arr_template_onehot, axis=-1)
            arr_template_seg = self._undo_one_hot(arr_template_seg, lut=self.synthseg_lut)

            save_volume(
                arr_template_seg,
                template_v2r,
                sss_seg_filepath,
            )

        return ProcessResult(
            exit_code=3,
            message="[done] subject space created. \n",
        )

    def _resample_to_session_space(
        self, sess_df: pd.DataFrame
    ) -> ProcessResult:
        """Resample all image modalities to a common session space

        Parameters
        ----------
        sess_df : pd.DataFrame
            Table with index_image as index and (subject, session_id, orig_mri, orig_synthseg) as columns.

        Returns
        -------
        ProcessResult
            ``{'exit_code': int, 'message': str, 'data': dict}``.
            Exit codes:
                ``-1`` error,
                ``0`` process is already completed (or has a single or no timpeoints available)
        """

        sss_kwargs = self.template_entities.copy()
        sss_kwargs["subject"] = sess_df["subject"].iloc[0]
        sss_kwargs["session"] = sess_df["session_id"].iloc[0]

        sss_file = self._get_data(**sss_kwargs, verbose=False)

        if sss_file is None:
            return ProcessResult(
                exit_code=-1, message="[error] Session-space has not been created.\n"
            )

        sss_proxy = nib.load(sss_file.path)
        for im_idx, im_files in sess_df.iterrows():
            im_file = im_files["orig_mri"]
            synthseg_file = im_files['orig_synthseg']

            extra_kwargs = {
                "subject": im_files['subject'],
                "session": im_files["session_id"],
                "run": self._get_entities(im_files['orig_synthseg']).get('run', None),
            }

            aff_file = self._get_data(
                **{**extra_kwargs,
                   **self.aff_graph_entities,
                   'suffix': self._get_entities(synthseg_file)['suffix'].split('synthseg')[0] + 'aff',
                   },
                verbose=False,
            )

            im_fname = self.build_path({
                **extra_kwargs,
                **self.im_session_entities,
                'suffix': self._get_entities(im_file)['suffix']
            })

            if aff_file is None:
                return ProcessResult(
                    exit_code=-1,
                    message="[error] could not find the affine file from session space to image " + synthseg_file +
                     ".\n",
                )

            aff = np.load(aff_file)
            if np.sum(np.isnan(aff)) > 0:
                return ProcessResult(
                    exit_code=-1,
                    message="[error] the affine matrix from session space to image " + synthseg_file + "has some"
                    "NaN values.\n",
                )

            im_proxy = nib.load(im_file)
            voxsize = np.sqrt(np.sum(im_proxy.affine * im_proxy.affine, axis=0))[:-1]
            voxsize_new = np.sqrt(np.sum(sss_proxy.affine * sss_proxy.affine, axis=0))[
                :-1
            ]
            factor = voxsize / voxsize_new
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling

            im_array = np.array(im_proxy.dataobj)
            im_array = gaussian_filter(im_array, sigmas)
            im_proxy = nib.Nifti1Image(im_array, np.linalg.inv(aff) @ im_proxy.affine)
            im_proxy = vol_resample_fast(sss_proxy, im_proxy)

            nib.save(im_proxy, join(DIR_PIPELINES[self.pipeline_dir], im_fname))


        return ProcessResult(exit_code=4,
                             message="[partly done] resampling to subject space correctly; "
                                     "missing registration to MNI\n")

    def _register_to_MNI_session(
            self, sess_df: pd.DataFrame
    ) -> ProcessResult:
        """Register the session-space T1w template to MNI and propagate to all modalities.

        Uses centroid-based affine alignment (same approach as
        :meth:`USLR_Linear._register_to_MNI`) with the session-space T1w
        template produced by :meth:`_create_session_space` as the moving image.
        The resulting affine is applied to every modality in session space and
        the resampled images are saved alongside the session-space outputs in
        ``nicgiprep-mm/sub-<subject>/ses-<session>/anat/``.

        Parameters
        ----------
        sess_df : pd.DataFrame
            Table with index_image as index and (subject, session_id, orig_mri, orig_synthseg) as columns.

        Returns
        -------
        ProcessResult
            ``{'exit_code': int, 'message': str, 'data': dict}``.
            Exit codes:
                ``-1`` error,
                ``0`` process is already completed (or has a single or no timpeoints available)
        """

        # extra = {"subject": subject, "session": session}

        # --- affine file ---
        extra_kwargs = {
            "subject": sess_df['subject'].iloc[0],
            "session": sess_df["session_id"].iloc[0],
        }

        mni_aff_entities = {
            "space": "MNI",
            "suffix": "aff",
            "extension": ".npy",
        }
        mni_aff_fname = self.build_path({**extra_kwargs, **mni_aff_entities})
        mni_aff_fpath = join(DIR_PIPELINES[self.pipeline_dir], mni_aff_fname)

        # Load the session-space template segmentation.
        sss_kwargs = self.template_entities.copy()
        sss_kwargs["subject"] = sess_df["subject"].iloc[0]
        sss_kwargs["session"] = sess_df["session_id"].iloc[0]
        sss_kwargs["suffix"] = "T1wsynthseg"

        template_seg_file = self._get_data(**sss_kwargs, verbose=False)
        if template_seg_file is None:
            return ProcessResult(
                exit_code=-1,
                message="[error] session-space T1w template segmentation not found; skipping MNI registration.\n",
            )

        # Centroid-based affine: MNI template → session-space T1w.
        centroid_ref, ok_ref = compute_centroids_ras(
            MNI_TEMPLATE_SEG, labels_registration
        )
        centroid_flo, ok_flo = compute_centroids_ras(
            template_seg_file.path, labels_registration
        )
        ok = (ok_ref > 0) & (ok_flo > 0)
        M_sbj = getM(centroid_ref[:, ok], centroid_flo[:, ok], use_L1=False)

        create_dir(dirname(mni_aff_fpath))
        np.save(mni_aff_fpath, M_sbj)

        return ProcessResult(
            exit_code=0,
            message="[done] registration to MNI completed. Subject is processed\n",
        )

    # ------------------------------------------------------------------ #
    #  Orchestration                                                      #
    # ------------------------------------------------------------------ #
    def process_session(
        self, sess_df: pd.DataFrame, force_flag: bool = False, **kwargs
    ) -> ProcessResult | None:
        """Run the full multimodal pipeline for a single session.

        Three phases:

        1. **Cross-sectional step** — handled upstream by
           :class:`MultiModalSynthSegProcessor`; this method reads from
           ``nicgiprep-cross``.
        2. **Joint multimodal registration** — COG centring, pairwise centroid
           SVD, spanning-tree solve, and session-space construction. Outputs go
           to ``nicgiprep-mm/sub-<subject>/ses-<session>/anat/``.
        3. **MNI registration** — centroid-based affine from the session-space
           T1w template to MNI; transform propagated to all modalities. Outputs
           saved alongside step 2 in ``nicgiprep-mm/.../anat/``.

        Parameters
        ----------
        sess_df : pd.DataFrame
            Table with session_id as index and orig_mri and orig_synthseg as columns
        force_flag : bool, optional
            If ``True``, rerun all steps. Default is ``False``.
        **kwargs
            Forwarded to :meth:`_solve_graph`.

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}`` checkpoint dict.
        """

        subject = sess_df.subject.iloc[0]
        session = sess_df['session_id'].iloc[0]

        sess_df.set_index("index_image", inplace=True, drop=False)


        def_dir = join(self.tmp_dir, "sub-" + subject, "ses-" + str(session))
        create_dir(def_dir)

        checkpoint = self._check_running_session(sess_df, force_flag)
        if checkpoint["exit_code"] in [-1, 0]:
            if kwargs.get("verbose", False):
                print(
                    "  * Session "
                    + str(session)
                    + " (files: "
                    + ", ".join([i.filename for i in sess_df['orig_synthseg']])
                    + ") -->",
                    checkpoint["message"],
                    end="",
                )
            return checkpoint

        # Step 1: Solve the session graph.
        if checkpoint["exit_code"] in [1]:
            def_dir = join(self.tmp_dir, "sub-" + subject, "ses-" + str(session))
            create_dir(def_dir)

            # Step 1a: Pairwise rigid registration (centroid SVD).
            t_cog_d = self._init_graph(sess_df, def_dir, force_flag)

            # Step 1b: Solve the log-space spanning tree (reuses USLR's solver).
            graph_kwargs = {
                "n_epochs": 30,
                "cost": "l1",
                "lr": 0.1,
                "max_iter": 20,
            }
            graph_kwargs.update(kwargs)

            checkpoint = self._solve_graph(subject, sess_df, def_dir, t_cog_d, **graph_kwargs)
            self._update_subject_layout(subject)


        # Step 2: Build the unbiased session space.
        if checkpoint["exit_code"] in [1, 2]:
            checkpoint = self._create_session_space(sess_df)
            self._update_subject_layout(subject)


        # Step 3: Resample all modalities to the session-space (empty).
        if checkpoint["exit_code"] in [1, 2, 3]:
            checkpoint = self._resample_to_session_space(sess_df)
            self._update_subject_layout(subject)

        # Step 4: Register session-space T1w to MNI and propagate to all modalities.
        if checkpoint["exit_code"] in [1, 2, 3, 4]:
            checkpoint = self._register_to_MNI_session(sess_df)
            self._update_subject_layout(subject)

        remove_dir(def_dir)

        return checkpoint

    def process_subject(
        self,
        subject: str,
        force_flag: bool = False,
        session_list: Optional[List[str]] = None,
        **kwargs
    ) -> ProcessResult:
        """Run multimodal joint registration for every session of one subject.

        Parameters
        ----------
        subject : str
            Subject ID.
        force_flag : bool, optional
            If ``True``, rerun all steps. Default is ``False``.
        session_list : list of str, optional
            Restrict processing to these session IDs. If ``None``, all
            sessions for the subject are processed.
        **kwargs
            Forwarded to :meth:`process_session`.

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}``.
        """
        print("\nSubject: " + subject, end='\n')
        sessions = self._get_sessions(subject)
        if session_list is not None:
            sessions = [s for s in sessions if s in session_list]

        sess_df = self._get_process_image_file(subject, sessions=sessions)
        if sess_df.empty:
            return ProcessResult(exit_code=0, message="subject does not have sessions with multimodal imaging available")

        sess_df.set_index("session_id", drop=False, inplace=True)

        sessions_failed = []
        for session in sessions:
            try:
                if session not in sess_df['session_id']: continue # no multimodal data for this session.
                ret_code = self.process_session(sess_df.loc[session], force_flag=force_flag, **kwargs)
                if ret_code['exit_code'] != 0:
                    if kwargs.get("verbose", False):
                        print(ret_code['message'])

            except Exception as e:
                if kwargs.get("verbose", False):
                    print(traceback.format_exc())
                sessions_failed += [session]

        if len(sessions_failed) > 0:
            if len(sessions_failed) == len(sessions):
                return ProcessResult(exit_code=1, message="subject has completely failed")
            else:
                return ProcessResult(exit_code=2, message="subject has partially failed; only "
                                                          + str(len(sessions) - len(sessions_failed)) +
                                                          ' of ' + str(len(sessions)) + ' could run.')

        return ProcessResult(exit_code=0, message="success")
