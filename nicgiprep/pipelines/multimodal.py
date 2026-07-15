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
from glob import glob
from os import makedirs
from os.path import join, dirname, basename, exists
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib
import pandas as pd
from bids.layout import BIDSFile
from scipy.ndimage import gaussian_filter
from skimage.morphology import ball, binary_dilation

from setup import *
from nicgiprep.pipelines.base import Processor
from nicgiprep.pipelines.cross_sectional import SynthSegProcessor
from nicgiprep.pipelines.longitudinal import USLRLinear
from nicgiprep.utils.io_utils import create_dir, save_volume, remove_dir
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
from nicgiprep.utils.label_utils import SYNTHSEG_LUT, SYNTHSEG_GMM_ONTOLOGY


class MMProcessor(Processor):
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

    #: Anatomical MRI contrasts handled by the multimodal pipeline.
    DEFAULT_MODALITIES = ["T1w", "T2w", "FLAIR", "PDw"]

    #: Modality elected as the session template.
    TEMPLATE_MODALITY = "T1w"

    #: BIDS ``space`` entity label for all common-space derivatives.
    SPACE = "session"

    #: Derivatives directory key / pybids scope for the multimodal outputs.
    PIPELINE_DIR = "nicgiprep-mm"

    #: Derivatives directory key for the cross-sectional preprocessing outputs.
    CROSS_PIPELINE_DIR = "nicgiprep-cross"

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
        self.aff_graph_entities = {
            "space": self.SPACE,
            "suffix": "aff",
            "extension": ".npy",
        }

        # Per-modality image / mask / seg resampled into the session space.
        self.im_graph_entities = {"space": self.SPACE, "extension": "nii.gz"}
        self.mask_graph_entities = {"space": self.SPACE, "extension": "nii.gz"}

        # Session template (S0): the aligned template-modality image + derivatives.
        self.template_entities = {
            "space": self.SPACE,
            "desc": "template",
            "extension": "nii.gz",
        }

        # Session-space (network) voxel-to-RAS array.
        self.net_v2r_entities = {
            "space": self.SPACE,
            "desc": "template",
            "suffix": "v2r",
            "extension": ".npy",
        }

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
            "scope": self.CROSS_PIPELINE_DIR,
            "extension": "nii.gz",
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
            "scope": self.CROSS_PIPELINE_DIR,
            "extension": "nii.gz",
            "suffix": self._seg_suffix(modality),
        }

    def _aff_desc(self, modality: str) -> str:
        """Build the ``desc`` value encoding a modality-to-template affine.

        Parameters
        ----------
        modality : str
            Modality suffix (e.g. ``'T1w'``).

        Returns
        -------
        str
            The ``desc`` entity value, e.g. ``'T1wtosession'``.
        """
        return modality + "to" + self.SPACE

    def _modality_aff_entities(self, modality: str) -> Dict:
        """Build BIDS entities for a modality's modality-to-template affine file.

        Parameters
        ----------
        modality : str
            Modality suffix (e.g. ``'T1w'``).

        Returns
        -------
        dict
            BIDS entity filters for the modality-to-template affine ``.npy``.
        """
        return {**self.aff_graph_entities, "desc": self._aff_desc(modality)}

    # ------------------------------------------------------------------ #
    #  Discovery helpers                                                  #
    # ------------------------------------------------------------------ #
    def _get_sessions(self, subject: str) -> List[str]:
        """Return all session IDs available for a subject.

        Parameters
        ----------
        subject : str
            Subject ID.

        Returns
        -------
        list of str
            Session IDs for the subject.
        """
        return self.bids_loader.get_session(subject=subject)

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


class MultiModalSynthSegProcessor(SynthSegProcessor):
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
        self.modalities = kwargs.get("modalities", list(MMProcessor.DEFAULT_MODALITIES))

    def _select_image(
        self, subject: str, session: str, modality: str
    ) -> Optional[BIDSFile]:
        """Select a single raw image of a given modality for a session.

        Generalises :meth:`SynthSegProcessor._select_image` (which is hardwired
        to ``T1w``) to an arbitrary modality suffix. When several candidate
        files exist, prefers files without an ``acquisition`` entity, then
        ``run-01``.

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.
        modality : str
            Modality suffix (e.g. ``'T2w'``).

        Returns
        -------
        bids.layout.BIDSFile or None
            The selected raw image, or ``None`` if none is found.
        """
        im_list = self._get_data(
            subject=subject,
            session=session,
            suffix=modality,
            extension=["nii", "nii.gz"],
            acquisition=["orig", None],
            scope="raw",
            ignore_check=True,
        )

        if len(im_list) == 0:
            return None

        elif len(im_list) > 1:
            if any(["acquisition" not in f.entities.keys() for f in im_list]):
                im_list_r = list(
                    filter(lambda x: "acquisition" not in x.entities.keys(), im_list)
                )
            elif any(["run" in f.entities.keys() for f in im_list]):
                im_list_r = list(filter(lambda x: x.entities["run"] == "01", im_list))
            else:
                im_list_r = im_list
            im_file = im_list_r[0]

        else:
            im_file = im_list[0]

        return im_file

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
            DIR_PIPELINES[MMProcessor.CROSS_PIPELINE_DIR],
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
    ) -> Tuple[List, List, List, List, List]:
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
        tuple of list
            ``(input_files, res_files, output_files, vol_files,
            discarded_files)`` — one entry per (session, modality) queued.
        """
        input_files, res_files, output_files, vol_files, discarded_files = (
            [],
            [],
            [],
            [],
            [],
        )

        available_sessions = self.bids_loader.get_session(subject=subject)
        if session_list is not None:
            available_sessions = [s for s in available_sessions if s in session_list]

        for session in available_sessions:
            cross_anat_dir = join(
                DIR_PIPELINES[MMProcessor.CROSS_PIPELINE_DIR],
                "sub-" + subject,
                "ses-" + session,
                "anat",
            )
            cross_utils_dir = join(
                DIR_PIPELINES[MMProcessor.CROSS_PIPELINE_DIR],
                "sub-" + subject,
                "ses-" + session,
                "utils",
            )

            # Verify that the T1w cross-sectional pipeline has been run first.
            t1w_img_done = bool(glob(join(cross_anat_dir, "*T1w.nii.gz")))
            t1w_dseg_done = bool(glob(join(cross_anat_dir, "*T1wdseg.nii.gz")))
            if not (t1w_img_done and t1w_dseg_done):
                print(
                    "\n  [skip] sub-"
                    + subject
                    + "/ses-"
                    + session
                    + ": T1w cross-sectional outputs not found in "
                    + cross_anat_dir
                    + ". Please run the T1w cross-sectional pipeline first."
                )
                continue

            # Collect raw modalities available for this session (T1w included for logging).
            to_process = []
            session_log: List[Tuple[str, str]] = []

            t1w_file = self._select_image(
                subject, session, MMProcessor.TEMPLATE_MODALITY
            )
            if t1w_file is not None:
                session_log.append((MMProcessor.TEMPLATE_MODALITY, t1w_file.path))

            for modality in self.modalities:
                if modality == MMProcessor.TEMPLATE_MODALITY:
                    continue
                im_file = self._select_image(subject, session, modality)
                if im_file is not None:
                    to_process.append((modality, im_file))
                    session_log.append((modality, im_file.path))

            # Always write the session log so downstream steps know which raw
            # files were used, even when there is nothing new to segment.
            self._write_session_inputs_log(subject, session, session_log)

            if not to_process:
                continue

            for modality, im_file in to_process:
                output_dir = cross_utils_dir

                if not exists(output_dir):
                    makedirs(output_dir)

                im_entities = {
                    k: str(v)
                    for k, v in im_file.entities.items()
                    if k in filename_entities
                }
                im_entities["acquisition"] = "1"
                seg_entities = {**im_entities, "suffix": self._seg_suffix(modality)}

                anat_res = basename(self.build_path(im_entities))
                anat_seg = basename(self.build_path(seg_entities))
                anat_vols = anat_seg.replace("nii.gz", "tsv")

                if not exists(join(output_dir, anat_seg)) or force_flag:
                    im_proxy = nib.load(im_file.path)
                    run_code = self._check_file(im_proxy)
                    if run_code["run_flag"]:
                        input_files.append(im_file.path)
                        # SynthSeg's resampled output is a temporary artifact; send it to
                        # TMP_DIR so it never lands in nicgiprep-cross.
                        res_files.append(join(TMP_DIR, anat_res))
                        output_files.append(join(output_dir, anat_seg))
                        vol_files.append(join(output_dir, anat_vols))
                    else:
                        with open(
                            join(output_dir, modality + "_excluded_file.txt"), "w"
                        ) as f:
                            f.write(run_code["exit_message"])

        return input_files, res_files, output_files, vol_files, discarded_files

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


class MultiModalBiasCorrectionProcessor(MMProcessor):
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

    def _check_resampled_file(self, raw_file, resampled_entities: Dict) -> Dict:
        """Return the path to the 1 mm resampled image needed for bias-field estimation.

        SynthSeg writes its ``--resample`` output to ``TMP_DIR`` (not to
        ``nicgiprep-cross``).  If that temporary file is still present it is
        reused directly.  Otherwise the raw image is resampled to 1 mm and
        saved in ``TMP_DIR``.  Nothing is written to ``nicgiprep-cross``.

        Parameters
        ----------
        raw_file : BIDSFile
            Source raw image.
        resampled_entities : dict
            BIDS entities used to derive the expected filename.

        Returns
        -------
        dict
            ``{'exit_code': 0, 'filepath': str}`` on success, or
            ``{'exit_code': -1, 'message': str}`` on failure.
        """
        tmp_path = join(TMP_DIR, basename(self.build_path(resampled_entities)))
        if exists(tmp_path):
            return {"exit_code": 0, "filepath": tmp_path}

        proxyraw = nib.load(raw_file.path)
        pixdim = np.sqrt(np.sum(proxyraw.affine * proxyraw.affine, axis=0))[:-1]
        if all([np.abs(p - 1) < 0.01 for p in pixdim]):
            return {"exit_code": 0, "filepath": raw_file.path}
        if any([p < 0.01 for p in pixdim]):
            return {"exit_code": -1, "message": "some dimensions are wrong"}
        v, aff = rescale_voxel_size(
            np.array(proxyraw.dataobj), proxyraw.affine, [1, 1, 1]
        )
        save_volume(v, aff, None, tmp_path)
        return {"exit_code": 0, "filepath": tmp_path}

    def _posterior2generative_labelmap(
        self, seg: np.ndarray, lut: Dict = SYNTHSEG_LUT
    ) -> np.ndarray:
        """Convert a soft posterior segmentation to a hemisphere-unified label space.

        Identical to
        :meth:`BiasCorrectionProcessor._posterior2generative_labelmap`.

        Parameters
        ----------
        seg : np.ndarray
            Soft segmentation ``(*spatial, n_labels)``.
        lut : dict, optional
            Label-to-channel lookup table. Defaults to ``SYNTHSEG_LUT``.

        Returns
        -------
        np.ndarray
            Normalised posteriors ``(*spatial, n_unified_classes)``.
        """
        out_seg = np.zeros(seg.shape[:-1] + (len(SYNTHSEG_GMM_ONTOLOGY),))
        for it_lab, (_, lab_list) in enumerate(SYNTHSEG_GMM_ONTOLOGY.items()):
            for lab in lab_list:
                out_seg[..., it_lab] += seg[..., lut[lab]]
        out_seg = out_seg / (np.sum(out_seg, axis=-1, keepdims=True) + 1e-10)
        out_seg[np.isnan(out_seg)] = 0
        return out_seg

    def process_subject(
        self,
        subject: str,
        force_flag: bool = False,
        remove_wrong: bool = True,
        session_list: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """Run bias-field correction and WM normalisation for every modality.

        For each session and each modality that has a segmentation in
        ``nicgiprep-cross``:

        1. Derives a brain mask from the SynthSeg segmentation (CSF excluded).
        2. Uses the ``acq-1`` resampled image (written by SynthSeg) as the
           reference for computing the bias field.
        3. Applies the bias field and normalises the WM mean to 110.
        4. Saves the corrected image under the raw filename (``acq=None``) so
           that downstream queries with ``acquisition=[None, 'orig']`` find it
           unambiguously.

        Parameters
        ----------
        subject : str
            Subject ID.
        force_flag : bool, optional
            If ``True``, reprocess even when output already exists. Default
            is ``False``.
        remove_wrong : bool, optional
            If ``True``, print an error and skip when bias-field estimation
            fails rather than raising. Default is ``True``.
        session_list : list of str, optional
            Restrict processing to these session IDs. If ``None``, all
            sessions for the subject are processed.
        **kwargs
            Ignored; present for API compatibility.

        Returns
        -------
        dict
            ``{'exit_code': 0, 'message': 'success'}``.
        """
        print("\nSubject: " + subject)

        available_sessions = self.bids_loader.get_session(subject=subject)
        if session_list is not None:
            available_sessions = [s for s in available_sessions if s in session_list]

        for session in available_sessions:
            print("\n* Session: " + session, end=": ", flush=True)

            for modality in self.modalities:
                # T1w bias correction is handled by the cross-sectional pipeline.
                if modality == self.template_modality:
                    continue

                seg_file = self._get_data(
                    **{
                        "session": session,
                        "subject": subject,
                        **self._modality_seg_entities(modality),
                    },
                    verbose=False
                )
                if seg_file is None:
                    print(modality + " not available. Skipping. ", end="", flush=True)
                    continue

                output_dir = join(
                    DIR_PIPELINES[self.CROSS_PIPELINE_DIR],
                    "sub-" + subject,
                    "ses-" + session,
                    self._modality_datatype(modality),
                )

                # Find the raw image from which this session's segmentation was derived.
                seg_ents = self._get_entities(seg_file)
                seg_ents["extension"] = "nii.gz"
                raw_entities = {k: v for k, v in seg_ents.items() if k != "acquisition"}
                raw_entities["suffix"] = modality
                raw_entities["scope"] = "raw"
                raw_entities["acquisition"] = [None, "orig"]

                raw_file = self._get_data(**raw_entities)
                if raw_file is None:
                    continue

                output_filepath = join(output_dir, basename(raw_file.path))

                if exists(output_filepath) and not force_flag:
                    print(modality + ": already done. ", end="", flush=True)
                    continue

                proxyraw = nib.load(raw_file.path)
                proxyseg = nib.load(seg_file.path)

                # --- 1 mm resampled image (written by SynthSeg to TMP_DIR; created here if absent) ---
                resampled_entities = dict(
                    seg_ents
                )  # has acq='1', same run/session/subject
                resampled_entities["suffix"] = modality
                resampled_entities["scope"] = self.CROSS_PIPELINE_DIR
                resampled_entities["datatype"] = self._modality_datatype(modality)

                resampled_flag = self._check_resampled_file(
                    raw_file, resampled_entities
                )
                if resampled_flag["exit_code"] == -1:
                    print(resampled_flag["message"], end="", flush=True)
                    continue

                proxyres = nib.load(resampled_flag["filepath"])

                # --- Bias-field correction ---
                if not exists(output_filepath) or force_flag:
                    vox2ras0 = proxyres.affine
                    mri_acq = np.asarray(proxyres.dataobj).astype("float32")
                    mri_acq[np.isnan(mri_acq)] = 0

                    # Resample seg to 1 mm if voxel sizes differ.
                    pixdimim = np.sqrt(np.sum(proxyres.affine**2, axis=0))[:-1]
                    pixdimseg = np.sqrt(np.sum(proxyseg.affine**2, axis=0))[:-1]
                    if any(np.abs(pixdimseg - pixdimim) > 0.01):
                        proxyseg_res = vol_resample_fast(
                            proxyres, proxyseg, mode="nearest"
                        )
                    else:
                        proxyseg_res = proxyseg

                    seg_arr_res = np.array(proxyseg_res.dataobj)
                    soft_seg = one_hot_encoding(seg_arr_res, categories=SYNTHSEG_LUT)
                    soft_seg = self._posterior2generative_labelmap(
                        soft_seg, lut=SYNTHSEG_LUT
                    )

                    try:
                        mri_acq_corr, bias_field = bias_field_corr(
                            mri_acq,
                            soft_seg,
                            penalty=1,
                            VERBOSE=False,
                            filter_exceptions=True,
                        )
                    except Exception:
                        mri_acq_corr = None

                    if mri_acq_corr is None:
                        print(
                            "[error] bias field cannot be computed for "
                            + modality
                            + ". ",
                            end="",
                            flush=True,
                        )
                        continue

                    del soft_seg
                    mask_bf = seg_arr_res > 0
                    wm_mask = (seg_arr_res == 2) | (seg_arr_res == 41)
                    del seg_arr_res

                    vox2ras0_orig = proxyraw.affine
                    mri_acq_orig = np.asarray(proxyraw.dataobj).astype("float32")
                    mri_acq_orig[np.isnan(mri_acq_orig)] = 0
                    if mri_acq_orig.ndim > 3:
                        mri_acq_orig = mri_acq_orig[..., 0]

                    vox_size = np.linalg.norm(vox2ras0, 2, 0)[:3]
                    orig_vox_size = np.linalg.norm(vox2ras0_orig, 2, 0)[:3]

                    if all(v1 == v2 for v1, v2 in zip(vox_size, orig_vox_size)):
                        # Same resolution: correct and normalise in-place.
                        mask_dilated = binary_dilation(mask_bf, ball(3))
                        m = np.mean(mri_acq_corr[wm_mask])
                        mri_acq_corr = 110 * mri_acq_corr / m
                        mri_acq_corr *= mask_dilated
                        save_volume(
                            np.clip(mri_acq_corr, 0, 255).astype("uint8"),
                            proxyres.affine,
                            None,
                            output_filepath,
                        )
                    else:
                        # Different resolutions: propagate bias field back to original res.
                        bias_proxy = nib.Nifti1Image(bias_field, proxyres.affine)
                        bias_field_resize = vol_resample_fast(
                            proxyraw, bias_proxy, return_np=True
                        )

                        mask_proxy = nib.Nifti1Image(
                            mask_bf.astype("float"), proxyres.affine
                        )
                        mask_resize = (
                            vol_resample_fast(proxyraw, mask_proxy, return_np=True)
                            > 0.5
                        )

                        wm_proxy = nib.Nifti1Image(
                            wm_mask.astype("float"), proxyres.affine
                        )
                        wm_resize = (
                            vol_resample_fast(proxyraw, wm_proxy, return_np=True) > 0.5
                        )

                        mri_orig_corr = copy.copy(mri_acq_orig)
                        mri_orig_corr[mask_resize] /= bias_field_resize[mask_resize]

                        m = np.mean(mri_orig_corr[wm_resize])
                        mri_orig_corr = 110 * mri_orig_corr / m
                        mask_dilated = binary_dilation(mask_resize, ball(3))
                        mri_orig_corr[~mask_dilated] = 0

                        save_volume(
                            np.clip(mri_orig_corr, 0, 255).astype("uint8"),
                            proxyraw.affine,
                            None,
                            output_filepath,
                        )

                        del (
                            bias_field,
                            bias_field_resize,
                            mri_acq_orig,
                            mri_orig_corr,
                            mask_dilated,
                        )

                    print(modality + ": done. ", end="", flush=True)

        return {"exit_code": 0, "message": "success"}


class MultiMRIProcessor(MMProcessor):
    """Joint, unbiased rigid registration of multiple MRI contrasts.

    For every session, estimates one rigid transform per modality that maps it
    into a session-specific common space lying at the centre of all modalities,
    by solving the same log-space spanning-tree problem as
    :class:`~nicgiprep.pipelines.longitudinal.USLR_Linear`. The aligned
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
        self.pipeline_dir = self.PIPELINE_DIR
        self.tmp_dir = join(self.tmp_dir, "SessionSpace")
        create_dir(self.tmp_dir)

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                      #
    # ------------------------------------------------------------------ #
    def _check_running_session(
        self, subject: str, session: str, modalities: List[str], force_flag: bool
    ) -> Dict:
        """Determine the processing checkpoint for a single session.

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.
        modalities : list of str
            Modalities available for the session.
        force_flag : bool
            If ``True``, ignore existing outputs and rerun.

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}``. Exit codes:
            ``-1`` error, ``0`` run full pipeline, ``1`` skip (already done),
            ``5`` single modality (nothing to register, link as template).
        """
        # Nothing to register if there are no modalities.
        if len(modalities) == 0:
            return {
                "exit_code": 1,
                "message": "[done] no usable modalities found. Skipping.\n",
            }

        # A single modality cannot define a joint space: skip multimodal registration.
        if len(modalities) == 1:
            return {
                "exit_code": 1,
                "message": "[skip] only T1w available. Multimodal registration requires at least 2 modalities.\n",
            }

        # Already processed: a per-modality affine exists for every modality and
        # the session template is present.
        im_template = self._get_data(
            **{
                "subject": subject,
                "session": session,
                "suffix": self.template_modality,
                **self.im_graph_entities,
            },
            verbose=False
        )
        all_affs = all(
            self._get_data(
                **{
                    "subject": subject,
                    "session": session,
                    **self._modality_aff_entities(m),
                },
                verbose=False
            )
            is not None
            for m in modalities
        )
        if all_affs and im_template is not None and not force_flag:
            return {
                "exit_code": 1,
                "message": "[done] session already processed. "
                "Check the results in [..]/session_space/sub-"
                + subject
                + "/ses-"
                + str(session)
                + ".\n",
            }

        return {"exit_code": 0, "message": "running multimodal registration"}

    # ------------------------------------------------------------------ #
    #  Rigid graph estimation (parcellation-based, log-space spanning tree)
    # ------------------------------------------------------------------ #
    def _get_centroids(
        self, subject: str, session: str, modalities: List[str]
    ) -> Tuple[Dict, Dict]:
        """Compute RAS centroids for each modality's segmentation.

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.
        modalities : list of str
            Modalities to process.

        Returns
        -------
        centroid_dict : dict
            ``{modality: np.ndarray}`` of shape ``(3, N_labels)``.
        ok_dict : dict
            ``{modality: np.ndarray}`` binary flags per label.
        """
        centroid_dict = {}
        ok_dict = {}
        for modality in modalities:
            seg_file = self._get_data(
                **{
                    "subject": subject,
                    "session": session,
                    **self._modality_seg_entities(modality),
                }
            )
            centroid_dict[modality], ok_dict[modality] = compute_centroids_ras(
                seg_file.path, labels_registration
            )

        return centroid_dict, ok_dict

    def _cog_path(self, seg_file: BIDSFile, modality: str) -> str:
        """Build the COG ``.npy`` path next to a modality's segmentation file.

        Parameters
        ----------
        seg_file : bids.layout.BIDSFile
            The modality's segmentation file.
        modality : str
            Modality suffix (e.g. ``'T1w'``).

        Returns
        -------
        str
            Path to the centre-of-gravity ``.npy`` file.
        """
        return (
            seg_file.path.replace("nii.gz", "npy")
            .replace(self._seg_suffix(modality), "cog")
            .replace("dseg", "cog")
        )

    def _compute_cog(self, subject: str, session: str, modalities: List[str]) -> None:
        """Compute and save a centring-to-COG transform for each modality.

        Mirrors :meth:`USLR_Linear._compute_cog`: the centre-of-gravity is
        derived from the non-zero voxels of each modality's segmentation and
        saved as a 4x4 translation matrix in RAS mm next to the segmentation.

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.
        modalities : list of str
            Modalities to process.

        Returns
        -------
        None
        """
        for modality in modalities:
            seg_file = self._get_data(
                **{
                    "subject": subject,
                    "session": session,
                    **self._modality_seg_entities(modality),
                }
            )
            cog_path = self._cog_path(seg_file, modality)

            seg_proxy = nib.load(seg_file.path)
            seg_arr = np.array(seg_proxy.dataobj)
            aux = np.where(seg_arr > 0)
            i, j, k = np.median(aux[0]), np.median(aux[1]), np.median(aux[2])
            ras_cog = seg_proxy.affine @ np.array([i, j, k, 1])
            T_cog = np.eye(4)
            T_cog[:3, -1] = -ras_cog[:3]
            np.save(cog_path, T_cog.astype("float32"))

    def _register_pair(
        self,
        pairwise_centroids: Sequence[np.ndarray],
        affine_filepath: str,
        ok_centr: Optional[np.ndarray] = None,
    ) -> None:
        """Estimate and save a rigid transform from centroid correspondences via SVD.

        Identical procedure to :meth:`USLR_Linear._register_timepoints` (closed
        form rigid fit from corresponding centroids), applied here to a pair of
        modalities.

        Parameters
        ----------
        pairwise_centroids : sequence of np.ndarray
            ``(refCent, floCent)`` — each of shape ``(3, N_labels)``.
        affine_filepath : str
            Path where the resulting 4x4 affine is saved as ``.npy``.
        ok_centr : np.ndarray, optional
            Binary flag array selecting reliable centroids (1 = use). If
            ``None``, all centroids are used.

        Returns
        -------
        None
        """
        refCent, floCent = pairwise_centroids

        if ok_centr is not None:
            refCent = refCent[:, ok_centr > 0]
            floCent = floCent[:, ok_centr > 0]

        trans_ref = np.mean(refCent, axis=1, keepdims=True)
        trans_flo = np.mean(floCent, axis=1, keepdims=True)

        refCent_tx = refCent - trans_ref
        floCent_tx = floCent - trans_flo

        cov = refCent_tx @ floCent_tx.T
        u, s, vt = np.linalg.svd(cov)
        D = np.eye(3)
        if np.prod(np.diag(s)) < 0:
            D[-1, -1] = -1

        Q = vt.T @ D @ u.T
        Tr = np.eye(4)
        Tr[:3, 3] = -trans_ref.squeeze()

        Tf = np.eye(4)
        Tf[:3, 3] = trans_flo.squeeze()

        R = np.eye(4)
        R[:3, :3] = Q

        aff = Tf @ R @ Tr
        np.save(affine_filepath, aff)

    def _init_graph(
        self,
        subject: str,
        session: str,
        modalities: List[str],
        def_dir: str,
        force_flag: bool,
    ) -> None:
        """Compute pairwise rigid affines between all modalities via centroid SVD.

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.
        modalities : list of str
            Modalities to register.
        def_dir : str
            Directory where pairwise ``<ref>_to_<flo>.npy`` affine files are
            written.
        force_flag : bool
            If ``True``, recompute even when files exist.

        Returns
        -------
        None
        """
        centroid_dict, ok_dict = self._get_centroids(subject, session, modalities)

        # Centre every modality on its COG before pairwise registration.
        for modality in modalities:
            seg_file = self._get_data(
                **{
                    "subject": subject,
                    "session": session,
                    **self._modality_seg_entities(modality),
                }
            )
            T_cog = np.load(self._cog_path(seg_file, modality))
            centroid_dict[modality] = T_cog @ np.concatenate(
                [
                    centroid_dict[modality],
                    np.ones((1, centroid_dict[modality].shape[1])),
                ]
            )
            centroid_dict[modality] = centroid_dict[modality][:3]

        for mod_ref, mod_flo in itertools.combinations(modalities, 2):
            output_filepath = join(
                def_dir, str(mod_ref) + "_to_" + str(mod_flo) + ".npy"
            )
            if not exists(output_filepath) or force_flag:
                self._register_pair(
                    [centroid_dict[mod_ref], centroid_dict[mod_flo]],
                    output_filepath,
                    ok_centr=(ok_dict[mod_ref] == 1) & (ok_dict[mod_flo] == 1),
                )

    def _solve_graph(
        self, subject: str, session: str, modalities: List[str], def_dir: str, **kwargs
    ) -> Dict:
        """Solve the rigid spanning-tree problem and save per-modality affines.

        Reuses the USLR log-space solver verbatim
        (:meth:`USLR_Linear.init_st2_lineal` and
        :meth:`USLR_Linear.st2_lineal_pytorch`), since the multimodal and USLR
        rigid models are mathematically identical — only the graph vertices differ
        (modalities here vs. timepoints in USLR).

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.
        modalities : list of str
            Modalities (graph vertices).
        def_dir : str
            Directory containing pairwise ``<ref>_to_<flo>.npy`` files.
        **kwargs
            Forwarded to :meth:`USLR_Linear.st2_lineal_pytorch`
            (e.g. ``n_epochs``, ``cost``, ``lr``).

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}`` checkpoint dict.
        """
        R_log = USLRLinear.init_st2_lineal(modalities, def_dir)
        Tres = USLRLinear.st2_lineal_pytorch(
            R_log, modalities, verbose=False, **kwargs
        )

        if np.sum(np.isnan(Tres)) > 0:
            return {
                "exit_code": -1,
                "message": "[error] Something went wrong in the rigid registration step.\n",
            }

        for it_mod, modality in enumerate(modalities):
            seg_file = self._get_data(
                **{
                    "subject": subject,
                    "session": session,
                    **self._modality_seg_entities(modality),
                }
            )
            filename = self.build_path(
                {
                    "subject": subject,
                    "session": session,
                    **self._modality_aff_entities(modality),
                }
            )

            affine_matrix = Tres[..., it_mod]
            T_cog = np.load(self._cog_path(seg_file, modality))

            output_filepath = join(DIR_PIPELINES[self.pipeline_dir], filename)
            create_dir(dirname(output_filepath))
            np.save(output_filepath, np.linalg.inv(T_cog) @ affine_matrix)

        return {
            "exit_code": 0,
            "message": "[partly done] graph computed; building session space.\n",
        }

    # ------------------------------------------------------------------ #
    #  Session space construction                                         #
    # ------------------------------------------------------------------ #
    def _create_session_space(
        self, subject: str, session: str, modalities: List[str]
    ) -> Dict:
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
        subject : str
            Subject ID.
        session : str
            Session ID.
        modalities : list of str
            Modalities to resample.

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}`` checkpoint dict.
        """
        extra = {"subject": subject, "session": session}

        # Gather solved affines and dilated masks in the common frame.
        aff_dict = {}
        mask_dilated_proxy_dict = {}
        for modality in modalities:
            aff_filename = self.build_path(
                {**extra, **self._modality_aff_entities(modality)}
            )
            aff = np.load(join(DIR_PIPELINES[self.pipeline_dir], aff_filename))
            seg_file = self._get_data(
                **{**extra, **self._modality_seg_entities(modality)}
            )
            if aff is None or seg_file is None:
                continue
            if np.sum(np.isnan(aff)) > 0:
                return {
                    "exit_code": -1,
                    "message": "[error] Something went wrong in the rigid registration step.\n",
                }

            aff_dict[modality] = aff

            seg_proxy = nib.load(seg_file.path)
            seg_arr = np.array(seg_proxy.dataobj)
            mask_arr = (seg_arr > 0) & (seg_arr != 24)

            mask_dilated_arr = binary_dilation(mask_arr, ball(3)).astype("float")
            mask_dilated_proxy_dict[modality] = nib.Nifti1Image(
                mask_dilated_arr, np.linalg.inv(aff) @ seg_proxy.affine
            )

        if len(mask_dilated_proxy_dict) == 0:
            return {
                "exit_code": -1,
                "message": "[error] no masks available to build the session space.\n",
            }

        # Define the common (network) space centred on the union bounding box.
        _, template_v2r, template_size = create_empty_template(
            list(mask_dilated_proxy_dict.values())
        )
        # tmp_template = join(
        #     self.tmp_dir, subject + "_" + str(session) + "_template.nii.gz"
        # )
        # save_volume(np.zeros(template_size), template_vox2ras0, None, tmp_template)
        #
        # ref_geom = sf.load_volume(tmp_template)
        # net2vox, vox2net, net_v2r = network_space(
        #     ref_geom, shape=self.net_shape, center=ref_geom
        # )
        ref_proxy = nib.Nifti1Image(np.zeros(template_size), template_v2r)

        filename_t_v2r = self.build_path({**extra, **self.net_v2r_entities})
        create_dir(dirname(join(DIR_PIPELINES[self.pipeline_dir], filename_t_v2r)))
        np.save(join(DIR_PIPELINES[self.pipeline_dir], filename_t_v2r), template_v2r)
        # os.remove(tmp_template)

        # Resample every modality into the session space.
        for modality in modalities:
            if modality not in aff_dict:
                continue
            aff = aff_dict[modality]

            im_file = self._get_data(
                **{**extra, **self._modality_image_entities(modality)}
            )
            seg_file = self._get_data(
                **{**extra, **self._modality_seg_entities(modality)}
            )
            if im_file is None or seg_file is None:
                continue

            # Anti-alias to 1 mm before resampling (don't blur if upsampling).
            im_proxy = nib.load(im_file.path)
            pixdim = np.sqrt(np.sum(im_proxy.affine * im_proxy.affine, axis=0))[:-1]
            factor = pixdim / np.array([1, 1, 1])
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0

            im_arr = np.array(im_proxy.dataobj)
            im_arr = gaussian_filter(im_arr, sigmas)
            im_proxy = nib.Nifti1Image(im_arr, np.linalg.inv(aff) @ im_proxy.affine)
            im_proxy = vol_resample_fast(ref_proxy, im_proxy)

            filename_im = self.build_path(
                {**extra, **self.im_graph_entities, "suffix": modality}
            )
            nib.save(im_proxy, join(DIR_PIPELINES[self.pipeline_dir], filename_im))

            if modality == self.template_modality:
                # Save T1w SynthSeg segmentation in session space — used by the
                # MNI registration step to compute centroid-based alignment.
                seg_proxy = nib.load(seg_file.path)
                seg_arr = np.array(seg_proxy.dataobj)
                seg_proxy = nib.Nifti1Image(
                    seg_arr, np.linalg.inv(aff) @ seg_proxy.affine
                )
                seg_proxy = vol_resample_fast(ref_proxy, seg_proxy, mode="nearest")
                filename_synthseg = self.build_path(
                    {
                        **extra,
                        **self.im_graph_entities,
                        "suffix": self._seg_suffix(modality),
                    }
                )
                nib.save(
                    seg_proxy, join(DIR_PIPELINES[self.pipeline_dir], filename_synthseg)
                )

                # Save T1w SuperSynth (dseg) segmentation in session space — final output.
                t1w_dseg_file = self._get_data(
                    **{
                        **extra,
                        "scope": self.CROSS_PIPELINE_DIR,
                        "extension": "nii.gz",
                        "suffix": "T1wdseg",
                        "datatype": "anat",
                    },
                    verbose=False
                )
                if t1w_dseg_file is not None:
                    t1w_dseg_proxy = nib.load(t1w_dseg_file.path)
                    t1w_dseg_arr = np.array(t1w_dseg_proxy.dataobj)
                    t1w_dseg_proxy = nib.Nifti1Image(
                        t1w_dseg_arr, np.linalg.inv(aff) @ t1w_dseg_proxy.affine
                    )
                    t1w_dseg_proxy = vol_resample_fast(
                        ref_proxy, t1w_dseg_proxy, mode="nearest"
                    )
                    filename_dseg = self.build_path(
                        {**extra, **self.im_graph_entities, "suffix": "T1wdseg"}
                    )
                    nib.save(
                        t1w_dseg_proxy,
                        join(DIR_PIPELINES[self.pipeline_dir], filename_dseg),
                    )
            # For non-T1w modalities: only the image is saved; SynthSeg segmentations
            # are used solely for registration and are not propagated to the output.

        return {"exit_code": 0, "message": "[done] session space created.\n"}

    def _link_single_modality(self, subject: str, session: str, modality: str) -> None:
        """Handle the degenerate single-modality session.

        With a single modality there is nothing to register: the modality is
        rescaled to 1 mm (if needed) and stored as the session template ``S0``
        with an identity modality-to-template affine.

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.
        modality : str
            The only available modality.

        Returns
        -------
        None
        """
        extra = {"subject": subject, "session": session}

        aff_fname = self.build_path({**extra, **self._modality_aff_entities(modality)})
        aff_fpath = join(DIR_PIPELINES[self.pipeline_dir], aff_fname)
        if not exists(aff_fpath):
            if not exists(dirname(aff_fpath)):
                makedirs(dirname(aff_fpath))
            np.save(aff_fpath, np.eye(4))

        im_fname = self.build_path(
            {**extra, **self.im_graph_entities, "suffix": modality}
        )
        im_fpath = join(DIR_PIPELINES[self.pipeline_dir], im_fname)
        if not exists(im_fpath):
            if not exists(dirname(im_fpath)):
                makedirs(dirname(im_fpath))
            im_file = self._get_data(
                **{**extra, **self._modality_image_entities(modality)}
            )
            if im_file is None:
                return

            im_proxy = nib.load(im_file.path)
            pixdim = np.sqrt(np.sum(im_proxy.affine * im_proxy.affine, axis=0))[:-1]
            if all([np.abs(p - 1) < 0.01 for p in pixdim]):
                rf = subprocess.call(
                    ["ln", "-s", im_file.path, im_fpath], stderr=subprocess.PIPE
                )
                if rf != 0:
                    subprocess.call(["cp", im_file.path, im_fpath])
            else:
                if any([p < 0.01 for p in pixdim]):
                    return
                im_arr_resc, aff_resc = rescale_voxel_size(
                    np.array(im_proxy.dataobj), im_proxy.affine, [1, 1, 1]
                )
                save_volume(im_arr_resc.astype("float32"), aff_resc, None, im_fpath)

        seg_file = self._get_data(**{**extra, **self._modality_seg_entities(modality)})
        if seg_file is not None:
            seg_fname = self.build_path(
                {
                    **extra,
                    **self.im_graph_entities,
                    "suffix": self._seg_suffix(modality),
                }
            )
            seg_fpath = join(DIR_PIPELINES[self.pipeline_dir], seg_fname)
            if not exists(seg_fpath):
                if not exists(dirname(seg_fpath)):
                    makedirs(dirname(seg_fpath))
                subprocess.call(
                    ["ln", "-s", seg_file.path, seg_fpath], stderr=subprocess.PIPE
                )

    # ------------------------------------------------------------------ #
    #  MNI registration                                                  #
    # ------------------------------------------------------------------ #
    def _register_to_MNI_session(
        self, subject: str, session: str, modalities: List[str]
    ) -> Dict:
        """Register the session-space T1w template to MNI and propagate to all modalities.

        Uses centroid-based affine alignment (same approach as
        :meth:`USLR_Linear._register_to_MNI`) with the session-space T1w
        template produced by :meth:`_create_session_space` as the moving image.
        The resulting affine is applied to every modality in session space and
        the resampled images are saved alongside the session-space outputs in
        ``nicgiprep-mm/sub-<subject>/ses-<session>/anat/``.

        Parameters
        ----------
        subject : str
            Subject ID.
        session : str
            Session ID.
        modalities : list of str
            Modalities to propagate the MNI transform to.

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}`` checkpoint dict.
        """
        extra = {"subject": subject, "session": session}

        # --- affine file ---
        mni_aff_entities = {
            "space": "MNI",
            "desc": "tosession",
            "suffix": "aff",
            "extension": ".npy",
        }
        mni_aff_fname = self.build_path({**extra, **mni_aff_entities})
        mni_aff_fpath = join(DIR_PIPELINES[self.pipeline_dir], mni_aff_fname)

        # Load the session-space T1w template and its segmentation.
        template_im_file = self._get_data(
            **{**extra, **self.im_graph_entities, "suffix": self.template_modality},
            verbose=False
        )
        template_seg_file = self._get_data(
            **{
                **extra,
                **self.im_graph_entities,
                "suffix": self._seg_suffix(self.template_modality),
            },
            verbose=False
        )

        if template_im_file is None or template_seg_file is None:
            return {
                "exit_code": -1,
                "message": "[error] session-space T1w template not found; skipping MNI registration.\n",
            }

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

        proxytemplate_mni = nib.load(MNI_TEMPLATE)

        # Resample each session-space modality into MNI space.
        for modality in modalities:
            im_file = self._get_data(
                **{**extra, **self.im_graph_entities, "suffix": modality}, verbose=False
            )
            if im_file is None:
                continue

            # Build MNI output entities from the session-space file entities.
            file_entities = {
                k: str(v) for k, v in im_file.entities.items() if k in filename_entities
            }
            file_entities.pop("acquisition", None)
            file_entities.pop("desc", None)
            file_entities["space"] = "MNI"

            im_mni_fpath = join(
                DIR_PIPELINES[self.pipeline_dir], self.build_path(file_entities)
            )
            create_dir(dirname(im_mni_fpath))

            # Anti-alias to 1 mm then apply affine into MNI space.
            proxyim = nib.load(im_file.path)
            pixdim = np.sqrt(np.sum(proxyim.affine * proxyim.affine, axis=0))[:-1]
            factor = pixdim / np.array([1, 1, 1])
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling
            im_arr = np.array(proxyim.dataobj)
            im_arr = gaussian_filter(im_arr, sigmas)
            proxyim = nib.Nifti1Image(im_arr, np.linalg.inv(M_sbj) @ proxyim.affine)
            proxyim = vol_resample_fast(proxytemplate_mni, proxyim)
            nib.save(proxyim, im_mni_fpath)

            # For T1w only: propagate the SuperSynth (dseg) segmentation to MNI space.
            # SynthSeg segmentations (used only for registration) are not saved here.
            # Non-T1w modality segmentations are not saved in the multimodal output.
            if modality == self.template_modality:
                t1w_session_dseg_file = self._get_data(
                    **{**extra, **self.im_graph_entities, "suffix": "T1wdseg"},
                    verbose=False
                )
                if t1w_session_dseg_file is not None:
                    dseg_entities = {**file_entities, "suffix": "T1wdseg"}
                    dseg_mni_fpath = join(
                        DIR_PIPELINES[self.pipeline_dir], self.build_path(dseg_entities)
                    )
                    proxyseg = nib.load(t1w_session_dseg_file.path)
                    proxyseg = nib.Nifti1Image(
                        np.array(proxyseg.dataobj),
                        np.linalg.inv(M_sbj) @ proxyseg.affine,
                    )
                    proxyseg = vol_resample_fast(
                        proxytemplate_mni, proxyseg, mode="nearest"
                    )
                    nib.save(proxyseg, dseg_mni_fpath)

        return {"exit_code": 0, "message": "[done] MNI registration complete.\n"}

    # ------------------------------------------------------------------ #
    #  Orchestration                                                      #
    # ------------------------------------------------------------------ #
    def process_session(
        self, subject: str, session: str, force_flag: bool = False, **kwargs
    ) -> Dict:
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
        subject : str
            Subject ID.
        session : str
            Session ID.
        force_flag : bool, optional
            If ``True``, rerun all steps. Default is ``False``.
        **kwargs
            Forwarded to :meth:`_solve_graph`.

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}`` checkpoint dict.
        """
        modalities = self._get_modalities(subject, session)
        checkpoint = self._check_running_session(
            subject, session, modalities, force_flag
        )
        print(
            "  * Session "
            + str(session)
            + " (modalities: "
            + ", ".join(modalities)
            + ") -->",
            checkpoint["message"],
            end="",
        )

        if checkpoint["exit_code"] in [-1, 1]:
            return checkpoint

        def_dir = join(self.tmp_dir, "sub-" + subject, "ses-" + str(session))
        create_dir(def_dir)

        # Step 2a: Centre images on their COG.
        self._compute_cog(subject, session, modalities)
        self._update_subject_layout(subject)

        # Step 2b: Pairwise rigid registration (centroid SVD).
        self._init_graph(subject, session, modalities, def_dir, force_flag)

        # Step 2c: Solve the log-space spanning tree (reuses USLR's solver).
        graph_kwargs = {
            "n_epochs": 30,
            "cost": "l1",
            "lr": 0.1,
            "dir_results": def_dir,
            "max_iter": 20,
        }
        graph_kwargs.update(kwargs)
        checkpoint = self._solve_graph(
            subject, session, modalities, def_dir, **graph_kwargs
        )
        self._update_subject_layout(subject)
        if checkpoint["exit_code"] == -1:
            return checkpoint

        # Step 2d: Build the unbiased session space and resample every modality.
        checkpoint = self._create_session_space(subject, session, modalities)
        self._update_subject_layout(subject)
        if checkpoint["exit_code"] == -1:
            return checkpoint

        remove_dir(def_dir)

        # Step 3: Register session-space T1w to MNI and propagate to all modalities.
        checkpoint = self._register_to_MNI_session(subject, session, modalities)
        print("  * MNI registration -->", checkpoint["message"], end="")

        # Remove the T1w SynthSeg segmentation that was saved in session space solely
        # to drive centroid-based MNI alignment; only the SuperSynth dseg is kept.
        synthseg_fname = self.build_path(
            {
                "subject": subject,
                "session": session,
                **self.im_graph_entities,
                "suffix": self._seg_suffix(self.template_modality),
            }
        )
        synthseg_fpath = join(DIR_PIPELINES[self.pipeline_dir], synthseg_fname)
        if exists(synthseg_fpath):
            os.remove(synthseg_fpath)

        self._update_subject_layout(subject)

        return checkpoint

    def process_subject(
        self,
        subject: str,
        force_flag: bool = False,
        session_list: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
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
        print("* Subject: " + subject)
        sessions = self._get_sessions(subject)
        if session_list is not None:
            sessions = [s for s in sessions if s in session_list]

        for session in sessions:
            try:
                self.process_session(subject, session, force_flag=force_flag, **kwargs)
            except Exception as e:
                print("    [error] session " + str(session) + " failed: " + str(e))

        print("DONE.\n")
        return {"exit_code": 1, "message": "success"}
