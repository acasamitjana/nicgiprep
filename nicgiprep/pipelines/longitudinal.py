"""
Processing pipeline classes for longitudinal neuroimaging data in BIDS format.

Provides base and specialised pipeline classes that wrap PyBIDS layout queries,
coordinate volumetric resampling, label fusion, and longitudinal volume tracking.
"""

import traceback
from typing import Optional, Literal
from os.path import join, dirname, exists
from joblib import delayed, Parallel
import itertools

from bids.layout import BIDSLayout
import torch
from torch import nn
from skimage.morphology import ball, binary_dilation
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
import numpy as np
import nibabel as nib
import pandas as pd
import surfa as sf


from setup import *
from nicgiprep.pipelines.base import Processor
from nicgiprep.models import InstanceRigidModelLOG
# from nicgiprep.callbacks import *
from nicgiprep.utils.preprocessing_utils import *
from nicgiprep.utils.label_utils import SYNTHSEG_APARC_LUT
from nicgiprep.utils.io_utils import create_dir, save_volume, ProcessResult
from nicgiprep.utils.synthmorph_utils import synthmorph_register, integrate_svf
from nicgiprep.utils.def_utils import vol_resample_fast, network_space, create_empty_template, compute_jacobian, getM
from nicgiprep.utils.fn_utils import one_hot_encoding,  compute_centroids_ras, gaussian_antialiasing



class LongitudinalProcessor(Processor):
    """Processing subclass that implement longitudinal-specific methods and attributes.
    It implements the USLR linear and nonlinear pipelines. It defines the outputs of each step

    Extends :class:`Processor` with entity dictionaries for the linear and
    nonlinear USLR registration outputs (affine graphs, SVFs, network-space
    images, and v2r arrays).
    """

    def _build_processor(self) -> None:
        """Extend the base processing entities with longitudinal entity definition.

        Adds the following attributes on top of those set by
        :meth:`Processor._build_processor`:

        - ``aff_long_ent`` — entities for subject-to-template affine files.
        - ``im_long_ent`` — entities for linearly registered images.
        - ``mask_long_ent`` — entities for brain masks in USLR space.
        - ``svf_long_ent`` — entities for nonlinear SVF graph files.
        - ``template_long_ent`` — entities for the linear template.
        - ``net_shape`` / ``svf_shape`` — default network and SVF spatial shapes.
        - ``net_v2r_ent`` / ``svf_v2r_ent`` — v2r affine file entities.
        """
        super()._build_processor()
        self.long_ent = {'space': 'subject', 'acquisition': '1', 'extension': 'nii.gz'}
        self.aff_long_ent = {'desc': 'raw2temp', 'suffix': 'aff', 'extension': '.npy'}
        self.im_long_ent = {'suffix': 'T1w', **self.long_ent}
        self.mask_long_ent = {'suffix': 'T1wmask', **self.long_ent}
        self.svf_long_ent = {'space': 'subject', 'suffix': 'svf', 'extension': 'nii.gz', 'datatype': 'utils'}
        self.template_long_ent = {'desc': 'template', 'suffix': 'T1w', **self.long_ent}

        self.net_shape = (192, 192, 192)
        self.svf_shape = (96, 96, 96)

        self.v2r_ent = {'datatype': 'utils', 'suffix': 'v2r', 'extension': '.npy', 'space': 'subject'}
        self.net_v2r_ent = {'desc': 'template', **self.v2r_ent}
        self.svf_v2r_ent = {'desc': 'svf', **self.v2r_ent}

        self.tmp_dir = join(self.tmp_dir, 'long')

        self.pipeline_dir = 'nicgiprep-long'


    def _name(self) -> str:
        """Return the display name of this pipeline."""
        return 'LongitudinalProcessor'


    def _get_sessions_file(self, subject: str) -> pd.DataFrame | ProcessResult:
        sess_fpath = join(DIR_PIPELINES[self.pipeline_dir], 'sub-' + subject, 'sub-' + subject + '_sessions.tsv')
        if exists(join(DIR_PIPELINES[self.pipeline_dir], 'sub-' + subject, 'sub-' + subject + '_sessions.tsv')):
            return pd.read_csv(sess_fpath, sep='\t')

        sess_df = {'session_id': [], 'orig_t1w': [], 'orig_seg': []}
        im_ent = {'scope': 'nicgiprep-cross', 'extension': 'nii.gz', 'suffix': 'T1w'}
        for sess_id in self.bids_loader.get_session(subject=subject):
            im_files = self.bids_loader.get(**{'subject': subject, 'session': sess_id, **im_ent})
            if len(im_files) == 0:
                return ProcessResult(exit_code=-1,
                                     message='[error] no T1w files are found for subject: ' + subject + ' and session '
                                             + str(sess_id) + '. Please run nicgiprep-cross first.')
            elif len(im_files) > 1:
                idx = np.random.choice(len(im_files), 1)
                im_file = im_files[idx]
            else:
                im_file = im_files[0]

            seg_ent = im_file.get_entities()
            seg_ent['suffix'] = ['T1wdseg', 'dseg']
            seg_files = self.bids_loader.get(**{'subject': subject, 'session': sess_id, **seg_ent})
            if len(im_files) == 0:
                return ProcessResult(exit_code=-1,
                                     message = '[error] no segmentation files are found for subject: ' + subject +
                                               ' and session ' + str(sess_id) + '. Please run nicgiprep-cross first.')

            seg_file = seg_files[0]

            sess_df['session_id'] += [sess_id]
            sess_df['orig_t1w'] += [im_file.path]
            sess_df['orig_seg'] += [seg_file.path]

        sess_df = pd.DataFrame(sess_df)
        sess_df.to_csv(sess_fpath, sep='\t', index=False)

        return sess_df

    def _get_session_time(self,
                          subject: str,
                          session_list: list[str],
                          session_df: Optional[pd.DataFrame]=None,
                          time_ref: Optional[str]=None) -> dict:
        """Build a mapping of session ID to time-from-baseline.

        Tries the columns ``'time_to_bl_days'``, ``'time_to_bl_years'``,
        and ``'age'`` in that order. Falls back to zero for all timepoints
        if none are found.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str
            Session IDs to include.
        session_df : pandas.DataFrame, optional
            Pre-loaded sessions DataFrame. Loaded from disk if ``None``.
        time_ref : string, optional
            String defining the column from sess_df that specifies time for list ordering.

        Returns
        -------
        dict
            Mapping ``{session_id: float}`` of time values.
        """
        session_map = {tp: 0 for tp in session_list}
        if session_df is None:
            session_df = self._get_subject_info(subject)

        if session_df is not None:
            candidates = (time_ref, 'time_to_bl_days', 'time_to_bl_years', 'age')
            time_ref = next((c for c in candidates if c in session_df.columns), None)
            if time_ref is None:
                return session_map

            session_map = {tp: float(session_df.loc[tp][time_ref]) for tp in session_list}

        return session_map

    def _get_last_tp(self, subject: str, session_list: list[str], time_mapping: Optional[dict]=None) -> str:
        """Return the session ID with the latest time value.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str
            Session IDs to include.
        time_mapping : dict, optional
            Pre-computed time mapping. Loaded via ``_get_session_time`` if ``None``.

        Returns
        -------
        str
            Session ID of the last timepoint available
        """
        if time_mapping is None:
            time_mapping = self._get_session_time(subject, session_list)

        tp_id = list(time_mapping.keys())
        tp_time = list(time_mapping.values())
        return tp_id[np.argmax(tp_time)]

    def _get_baseline_tp(self, subject: str, session_list: list[str], time_mapping: Optional[dict]=None) -> str:
        """Return the session ID with the earliest time value.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str
            Session IDs to include.
        time_mapping : dict, optional
            Pre-computed time mapping. Loaded via ``_get_session_time`` if ``None``.

        Returns
        -------
        str
            Session ID of the baseline timepoint.
        """
        if time_mapping is None:
            time_mapping = self._get_session_time(subject, session_list)

        tp_id = list(time_mapping.keys())
        tp_time = list(time_mapping.values())
        return tp_id[np.argmin(tp_time)]


class USLRLinear(LongitudinalProcessor):
    """Rigid longitudinal registration via the USLR spanning-tree algorithm.

    Estimates per-session rigid transforms to a latent (unknown) template by jointly minimising
    pairwise centroid-based rigid fitting losses (log-space Lie-algebra parameterisation).
    """

    def _name(self):
        """Return the display name of this pipeline."""
        return 'Longitudinal:Linear-Registration'

    def _build_processor(self):
        """Extend the base processor with linear-registration output entities."""
        super()._build_processor()
        self.tmp_dir = join(self.tmp_dir, 'long-lin-reg')
        create_dir(self.tmp_dir)
        self.pipeline_dir = 'nicgiprep-long'

    def _check_running_subject(self,
                               subject: str,
                               session_list: list[str],
                               force_flag: bool = False,
                               ) -> ProcessResult:

        """Determine the processing checkpoint for a subject.

        Parameters
        ----------
        subject : str
            Subject ID of the processing subject.
        session_list : list of str
            Session IDs to include in the processing.
        force_flag : bool, optional
            If ``True``, ignore existing outputs and rerun.

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}``.  Exit codes:
            ``-1`` error, ``0`` run full pipeline, ``1`` skip,
            ``2`` graph done,
            ``5`` single timepoint.
        """
        # do not run if only 1 timepoint available
        if len(session_list) == 1:
            return ProcessResult(exit_code=5, message='[done] It has only 1 timepoint. Linking files. \n')

        # do not run if only 0 timepoint available
        elif len(session_list) == 0:
            return ProcessResult(exit_code=1, message='[done] It has 0 sessions available. Skipping.\n')


        # do not run if it has already been processed
        elif (self._get_data(**{'subject': subject, **self.aff_long_ent}, curr_len=len(session_list), verbose=False)
              is not None and not force_flag):

           if not exists(join(DIR_PIPELINES[self.pipeline_dir], 'sub-' + subject, 'sub-' + subject + '_T1wetiv.npy')):
                return ProcessResult(exit_code=2, message='[partly done] graph is already computed;  etiv missing.\n')
           else:
                return ProcessResult(exit_code=1,
                                     message='[done] subject already processed. Check the results in [..]/'
                                             + self.pipeline_dir + '/sub-' + subject + '.\n')

        # need to entire pipeline
        else:
            return ProcessResult(exit_code=0, message='Subject needs to be processed')

    def _register_timepoints(self,
                             ref_centroid: np.ndarray,
                             flo_centroid: np.ndarray,
                             affine_filepath: str,
                             ok_centr: Optional[np.ndarray]=None,
                             force_flag: bool=False) -> None:
        """Estimate and save a rigid transforms between for every combination of T1w sessions
        from centroid correspondences via SVD.

        Reference: https://www.cse.sc.edu/~songwang/CourseProj/proj2004/ross/ross.pdf

        Parameters
        ----------
        ref_centroid : np.ndarray
            Centroids for the N ROIs in RAS coordinates (mm) from the reference image each of shape ``(3, N)``.
        flo_centroid : tuple of np.ndarray
            Centroids for the N ROIs in RAS coordinates (mm) from the floating image each of shape ``(3, N)``.
        affine_filepath : str
            Path where the resulting 4×4 affine is saved as ``.npy``.
        ok_centr : np.ndarray, optional
            Binary flag array selecting reliable centroids (1 = use). If ``None``, all centroids are used.
        force_flag : np.ndarray, optional, default to False
            Binary flag to indicate whether to re-compute the registration if it already exists
        """
        #
        if exists(affine_filepath) and not force_flag:
            return

        if ok_centr is not None:
            ref_centroid = ref_centroid[:, ok_centr > 0]
            flo_centroid = flo_centroid[:, ok_centr > 0]

        trans_ref = np.mean(ref_centroid, axis=1, keepdims=True)
        trans_flo = np.mean(flo_centroid, axis=1, keepdims=True)

        ref_cent_tx = ref_centroid - trans_ref
        flo_cent_tx = flo_centroid - trans_flo

        cov = ref_cent_tx @ flo_cent_tx.T
        u, s, vt = np.linalg.svd(cov)
        I = np.eye(3)
        if np.prod(np.diag(s)) < 0:
            I[-1, -1] = -1

        Q = vt.T @ I @ u.T

        # Full transformation
        Tr = np.eye(4)
        Tr[:3, 3] = -trans_ref.squeeze()

        Tf = np.eye(4)
        Tf[:3, 3] = trans_flo.squeeze()

        R = np.eye(4)
        R[:3, :3] = Q

        aff = Tf @ R @ Tr

        np.save(affine_filepath, aff)

    def _get_centroids(self, sess_df: pd.DataFrame):
        """Compute RAS centroids (mm) for each ROI segmented on all the available sessions .
        Each segmentation provides N_label ROIs.

        Parameters
        ----------
        sess_df : pd.DataFrame
            Dataframe with session_id as index and orig_t1w and orig_seg as columns

        Returns
        -------
        centroid_dict : dict
            ``{session_id: np.ndarray}`` of shape ``(3, N_labels)``.
        ok : dict
            ``{session_id: np.ndarray}`` binary flags per label.
        """
        centroid_dict = {}
        ok = {}
        for sess_id, sess_files in sess_df.iterrows():
            seg_filepath = sess_files['orig_seg']
            centroid_dict[sess_id], ok[sess_id] = compute_centroids_ras(seg_filepath, labels_registration)

        return centroid_dict, ok

    def _compute_cog(self, sess_df: pd.DataFrame) -> dict:
        """Compute and save a centring-to-COG transform for each session.

        The centre-of-gravity (COG) is computed from the non-zero voxels of
        each segmentation and saved as a 4×4 translation matrix in RAS mm.

        Parameters
        ----------
        sess_df : pd.DataFrame
            Dataframe with session_id as index and orig_t1w and orig_seg as columns
        """
        T_cog_d = {}
        for sess_id, sess_files in sess_df.iterrows():
            seg_filepath = sess_files['orig_seg']

            seg_proxy = nib.load(seg_filepath)
            data = np.array(seg_proxy.dataobj)
            aux = np.where(data>0)
            i, j, k = np.median(aux[0]), np.median(aux[1]), np.median(aux[2])
            ras_cog = seg_proxy.affine @ np.array([i, j, k, 1])
            T_cog = np.eye(4)
            T_cog[:3, -1] = -ras_cog[:3]
            T_cog_d[sess_id] = T_cog

        return T_cog_d

    def _init_graph(self,  sess_df: pd.DataFrame, def_dir: str, force_flag: bool=False) -> dict:
        """Compute pairwise rigid affines between all timepoints via centroid SVD.

        Parameters
        ----------
        sess_df : pd.DataFrame
            DataFrame with session_id as index and orig_t1w and orig_seg as columns
        def_dir : str
            Directory where pairwise ``.npy`` affine files are written.
        force_flag : bool
            If ``True``, recompute even when files exist.
        """
        # compute centroids
        t_cog_d = self._compute_cog(sess_df)
        centroids_dict, ok_dict = self._get_centroids(sess_df)

        for sess_id in sess_df.index:
            t_cog = t_cog_d[sess_id]
            centroids_dict[sess_id] = t_cog @ np.concatenate([centroids_dict[sess_id], np.ones((1, centroids_dict[sess_id].shape[1]))])
            centroids_dict[sess_id] = centroids_dict[sess_id][:3]

        # pairwise registration
        for sess_ref, sess_flo in itertools.combinations(sess_df.index, 2):
            output_filepath = join(def_dir, str(sess_ref) + '_to_' + str(sess_flo) + '.npy')
            ok_cent = (ok_dict[sess_ref] == 1) & (ok_dict[sess_flo] == 1)
            self._register_timepoints(centroids_dict[sess_ref], centroids_dict[sess_flo], output_filepath,
                                      ok_centr=ok_cent, force_flag=force_flag)

        return t_cog_d

    def _solve_graph(self, subject: str, sess_df: pd.DataFrame, def_dir: str, t_cog_d: dict, **kwargs) -> ProcessResult:
        """Solve the rigid spanning-tree problem and save per-timepoint affines.

        Reads pairwise log-rigid observations, fits the USLR model via
        L-BFGS, and writes one ``.npy`` affine per timepoint.

        Parameters
        ----------
        subject : str
            Subject ID.
        sess_df: pd.DataFrame
            DataFrame with session_id as index and orig_t1w and orig_seg as columns, def_dir: str, t_cog_d: dict, **kwargs) -> ProcessResult:

        def_dir : str
            Directory containing pairwise ``<ref>_to_<flo>.npy`` files.

        t_cog_d : dict[np.ndarray]
            Dictionary with COG for every session. (Keys=session ID, values=COG).

        **kwargs
            Forwarded to :meth:`st2_lineal_pytorch`
            (e.g. ``n_epochs``, ``cost``, ``lr``).

        Returns
        -------
        dict
            Checkpoint dict with ``exit_code=2``.
        """
        log_r = USLRLinear.init_st2_lineal(sess_df.index, def_dir)
        t_res = USLRLinear.st2_lineal_pytorch(log_r, sess_df.index, verbose=False, **kwargs)

        if np.sum(np.isnan(t_res)) > 0:
            return ProcessResult(exit_code=-1, message='[error] Something went wrong in the rigid registration step.\n')

        for it_sess_id, sess_id in enumerate(sess_df.index):

            extra_kwargs = {'session': sess_id, 'subject': subject}
            filename = self.build_path({**extra_kwargs, **self.aff_long_ent})

            affine_matrix = t_res[..., it_sess_id]
            T_cog = t_cog_d[sess_id]

            output_filepath = join(DIR_PIPELINES[self.pipeline_dir], filename)
            create_dir(dirname(output_filepath))

            np.save(output_filepath, np.linalg.inv(T_cog) @ affine_matrix)

        return ProcessResult(exit_code=0, message='[success]\n')

    def _create_subject_space(self, subject: str, sess_df: pd.DataFrame) -> ProcessResult|None:
        """Build a 1 mm isotropic network-space template for the subject.

        Computes an average bounding box from all timepoints' brain masks,
        defines a network space (LIA, 1 mm, 192³), resamples each session's
        image and segmentation there, takes the median image and majority-vote
        segmentation, and saves everything to ``uslr-lin``.

        Parameters
        ----------
        subject : str
            Subject ID.
        sess_df : pd.DataFrame
            DataFrame with session_id as index and orig_t1w and orig_seg as columns

        Returns
        -------
        dict
            Checkpoint dict with ``exit_code=3``.
        """
        masks, masks_dilated = [], []
        for sess_id, sess_files in sess_df.iterrows():
            aff_file = self._get_data(**{'session': sess_id, 'subject': subject, **self.aff_long_ent})
            seg_file = sess_files['orig_seg']

            if aff_file is None or seg_file is None:
                return ProcessResult(exit_code=-1,
                                     message='[error] Something went wrong in the rigid registration step.\n')

            aff = np.load(aff_file)
            if np.sum(np.isnan(aff)) > 0:
                return ProcessResult(exit_code=-1,
                                     message='[error] Something went wrong in the rigid registration step.\n')

            seg_proxy = nib.load(seg_file)
            seg_arr = np.array(seg_proxy.dataobj)
            mask_arr = (seg_arr > 0)
            mask_dilated_arr = binary_dilation(mask_arr, ball(3)).astype('float')
            masks_dilated.append(nib.Nifti1Image(mask_dilated_arr, np.linalg.inv(aff) @ seg_proxy.affine))

        # create subject space
        _, template_v2r, template_size = create_empty_template(masks_dilated, margin_bb=5)
        save_volume(np.zeros(template_size), template_v2r, join(self.tmp_dir, subject + '_template.nii.gz'))

        # move subject space to network space.
        # this is necessary because mri_synthmorph does not output the SVF files needed in this project.
        # thus, we already initialize subject space in the "network space" so that the longitudinal trajectories
        # already lie in the subject space.
        template = sf.load_volume(join(self.tmp_dir, subject + '_template.nii.gz'))
        net2vox, vox2net, net_v2r = network_space(template, shape=self.net_shape, center=template)
        svf_v2r = net_v2r.copy()
        for c in range(3):
            svf_v2r[:-1, c] = svf_v2r[:-1, c] / 0.5
        svf_v2r[:-1, -1] = svf_v2r[:-1, -1] - np.matmul(svf_v2r[:-1, :-1], 0.5 * (np.array([0.5] * 3) - 1))

        sss_kwargs = self.template_long_ent.copy()
        sss_kwargs['subject'] = 'subject'
        sss_kwargs['suffix'] = 'empty'
        sss_kwargs['datatype'] = 'utils'

        root_dir = DIR_PIPELINES[self.pipeline_dir]
        sss_filepath = join(root_dir, self.build_path({'subject': subject, **sss_kwargs}))
        sss_v2r_filepath = join(root_dir, self.build_path({'subject': subject, **self.net_v2r_ent}))
        svf_v2r_filepath = join(root_dir, self.build_path({'subject': subject, **self.svf_v2r_ent}))
        save_volume(np.zeros(self.net_shape), net_v2r, sss_filepath)
        np.save(sss_v2r_filepath, net_v2r)
        np.save(svf_v2r_filepath, svf_v2r)

        subprocess.call(['rm', '-rf', join(self.tmp_dir, subject + '_template.nii.gz')])

        return None

    def _resample_to_subject_space(self, subject: str, sess_df: pd.DataFrame) -> ProcessResult:
        """Estimate and save the total intra-cranial volume (eTIV) for a subject.

        Averages binary brain masks across timepoints in the network space and
        saves the resulting voxel count.

        Parameters
        ----------
        subject : str
            Subject ID.
        sess_df : pd.DataFrame
            DataFrame with session_id as index and orig_t1w and orig_seg as columns
        """

        sss_kwargs = self.template_long_ent.copy()
        sss_kwargs['subject'] = 'subject'
        sss_kwargs['suffix'] = 'empty'
        sss_kwargs['datatype'] = 'utils'

        root_dir = DIR_PIPELINES[self.pipeline_dir]
        sss_filepath = join(root_dir, self.build_path({'subject': subject, **sss_kwargs}))
        if not exists(sss_filepath):
            return ProcessResult(exit_code=-1, message='[error] Subject-space has not been created.\n')

        sss_proxy = nib.load(sss_filepath)
        for sess_id, sess_files in sess_df.iterrows():
            im_file = sess_files['orig_t1w']
            extra_kwargs = {'subject': subject, 'session': sess_id}
            aff_file = self._get_data(**{**extra_kwargs, **self.aff_long_ent})
            im_fname = self.build_path({**extra_kwargs, **self.im_long_ent})

            if aff_file is None:
                return ProcessResult(exit_code=-1, message='[error] Something went wrong in the rigid registration '
                                                           'step for' + str(extra_kwargs) + '.\n')

            aff = np.load(aff_file)
            if np.sum(np.isnan(aff)) > 0:
                return ProcessResult(exit_code=-1, message='[error] Something went wrong in the rigid registration '
                                                           'step for' + str(extra_kwargs) + '.\n')

            im_proxy = nib.load(im_file.path)
            voxsize = np.sqrt(np.sum(im_proxy.affine * im_proxy.affine, axis=0))[:-1]
            voxsize_new = np.sqrt(np.sum(sss_proxy.affine * sss_proxy.affine, axis=0))[:-1]
            factor = voxsize / voxsize_new
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling

            im_array = np.array(im_proxy.dataobj)
            im_array = gaussian_filter(im_array, sigmas)
            im_proxy = nib.Nifti1Image(im_array, np.linalg.inv(aff) @ im_proxy.affine)
            im_proxy = vol_resample_fast(sss_proxy, im_proxy)

            nib.save(im_proxy, join(DIR_PIPELINES[self.pipeline_dir], im_fname))

        return ProcessResult(exit_code=0, message='succeed')

    def _compute_etiv(self, subject: str, sess_df: pd.DataFrame) -> ProcessResult:
        """Estimate and save the total intra-cranial volume (eTIV) for a subject.

        Averages binary brain masks across timepoints in the network space and
        saves the resulting voxel count.

        Parameters
        ----------
        subject : str
            Subject ID.
        sess_df : pd.DataFrame
            DataFrame with session_id as index and orig_t1w and orig_seg as columns
        """

        sss_kwargs = self.template_long_ent.copy()
        sss_kwargs['subject'] = 'subject'
        sss_kwargs['suffix'] = 'empty'
        sss_kwargs['datatype'] = 'utils'

        root_dir = DIR_PIPELINES[self.pipeline_dir]
        sss_filepath = join(root_dir, self.build_path({'subject': subject, **sss_kwargs}))
        if not exists(sss_filepath):
            return ProcessResult(exit_code=-1, message='[error] Subject-space has not been created.\n')

        sss_proxy = nib.load(sss_filepath)
        template_mask = np.zeros(self.net_shape + (len(SYNTHSEG_APARC_LUT),))
        for sess_id, sess_files in sess_df.iterrows():
            extra_kwargs = {'subject': subject, 'session': sess_id}
            aff_file = self._get_data(**{**extra_kwargs, **self.aff_long_ent})
            seg_file = sess_files['orig_seg']

            if aff_file is None or seg_file is None:
                return ProcessResult(exit_code=-1, message='[error] Something went wrong in the rigid registration '
                                                           'step for' + str(extra_kwargs) + '.\n')

            aff = np.load(aff_file)
            if np.sum(np.isnan(aff)) > 0:
                return ProcessResult(exit_code=-1, message='[error] Something went wrong in the rigid registration '
                                                           'step for' + str(extra_kwargs) + '.\n')


            seg_proxy = nib.load(seg_file)
            seg_arr = np.array(seg_proxy.dataobj)
            mask_arr = (seg_arr > 0)
            mask_proxy = nib.Nifti1Image(mask_arr.astype('uint8'), np.linalg.inv(aff) @ seg_proxy.affine)
            template_mask += vol_resample_fast(sss_proxy, mask_proxy, return_np=True) / len(sess_df)

        etiv = np.sum(template_mask)
        etiv_path = self.build_path({'subject': subject, 'suffix': 'T1wetiv', 'extension':'npy'})
        np.save(join(DIR_PIPELINES[self.pipeline_dir], etiv_path), etiv)

        return ProcessResult(exit_code=0, message='succeed')

    def process_subject(self, subject: str, force_flag: bool=False, **kwargs) -> ProcessResult:
        """Run the full linear USLR pipeline for one subject.

        Orchestrates:
        * COG centering;
        * pairwise centroid registration;
        * spanning-tree solution;
        * and eTIV computation.

        Parameters
        ----------
        subject : str
            Subject ID.
        force_flag : bool, optional
            If ``True``, rerun all steps. Default is ``False``.
        register_MNI : bool, optional
            If ``True``, also register each session to MNI space. Default
            is ``False``.
        **kwargs
            Forwarded to :meth:`_solve_graph`.
        """
        exit_dict = ProcessResult(exit_code=0, message='success')
        try:
            def_dir = join(self.tmp_dir, 'sub-' + subject)
            create_dir(def_dir)

            sess_df = self._get_sessions_file(subject)
            if not isinstance(sess_df, pd.DataFrame):
                if kwargs.get('verbose', False): print(sess_df['message'])
                return sess_df

            session_list = sess_df['session_id']
            sess_df.set_index('session_id', drop=False, inplace=True)

            checkpoint = self._check_running_subject(subject, session_list, force_flag)

            if kwargs.get('verbose', False): print('* Subject: ' + subject)
            if checkpoint['exit_code'] == -1 or checkpoint['exit_code'] == 1 or checkpoint['exit_code'] == 5:
                if kwargs.get('verbose', False): print(checkpoint['message'])
                return checkpoint

            if checkpoint['exit_code'] in [0]:
                # initialize graph
                t_cog_d = self._init_graph(sess_df, def_dir, force_flag)
                self._update_subject_layout(subject)

                # compute graph
                tmp_dir = join(self.tmp_dir, subject)
                create_dir(tmp_dir)
                graph_kwargs = {'n_epochs': 30, 'cost': 'l1', 'lr': 0.1, 'dir_results': tmp_dir, 'max_iter': 20}
                self._solve_graph(subject, sess_df, def_dir, t_cog_d, **graph_kwargs)
                self._update_subject_layout(subject)

            if checkpoint['exit_code'] in [0, 2]:
                # create subject space
                pr = self._create_subject_space(subject, sess_df)
                if pr['exit_code'] != 0:
                    return pr

                pr = self._resample_to_subject_space(subject, sess_df)
                if pr['exit_code'] != 0:
                    return pr

                pr = self._compute_etiv(subject, sess_df)
                if pr['exit_code'] != 0:
                    return pr

                self._update_subject_layout(subject)

            return exit_dict

        except Exception as e:
            if kwargs.get('verbose', False): print(traceback.format_exc())
            return ProcessResult(exit_code=-1, message=f'[error] subject {subject} failed: {e}')

    @staticmethod
    def init_st2_lineal(session_list: list[object], input_dir: str, eps: float=1e-6):
        """Load pairwise rigid affines and compute their log-space representations.

        Reads ``<ref>_to_<flo>.npy`` files and returns the log-rigid
        (Euler angles + log-translation) for each pair.

        Parameters
        ----------
        session_list : list
            Session IDs to include (strings or objects with ``.id``).
        input_dir : str
            Directory containing the pairwise ``.npy`` affine files.
        eps : float, optional
            Numerical offset for safe ``arccos`` evaluation. Default is 1e-6.

        Returns
        -------
        np.ndarray
            Log-rigid observations, shape ``(6, K)`` where
            ``K = N*(N-1)//2``.
        """
        nk = 0

        N = len(session_list)
        K = int(N * (N - 1) / 2)

        phi_log = np.zeros((6, K))

        for sess_ref, sess_flo in itertools.combinations(session_list, 2):
            if not isinstance(sess_ref, str):
                tid_ref, tid_flo = sess_ref.id, sess_flo.id
            else:
                tid_ref, tid_flo = sess_ref, sess_flo

            filename = str(tid_ref) + '_to_' + str(tid_flo)

            rigid_matrix = np.load(join(input_dir, filename + '.npy'))
            rotation_matrix, translation_vector = rigid_matrix[:3, :3], rigid_matrix[:3, 3]

            # Log(R) and Log(T)
            t_norm = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1 + eps, 1 - eps)) + eps
            W = 1 / (2 * np.sin(t_norm)) * (rotation_matrix - rotation_matrix.T) * t_norm
            Vinv = np.eye(3) - 0.5 * W + ((1 - (t_norm * np.cos(t_norm / 2)) / (
                        2 * np.sin(t_norm / 2))) / t_norm ** 2) * W * W  # np.matmul(W, W)

            phi_log[0, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * t_norm
            phi_log[1, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * t_norm
            phi_log[2, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * t_norm

            phi_log[3:, nk] = np.matmul(Vinv, translation_vector)

            nk += 1

        return phi_log

    @staticmethod
    def st2_lineal_pytorch(logr: np.ndarray,
                           session_list: list[object],
                           n_epochs: int,
                           cost: Literal["l1", "l2"],
                           lr: float,
                           dir_results: str,
                           max_iter: int=5,
                           patience: int=3,
                           device: str='cpu',
                           verbose: bool=True):
        """Fit the rigid USLR model via L-BFGS optimisation.

        For exactly 2 timepoints, closes in closed form. For > 2, uses
        :class:`~nicgiprep.models.InstanceRigidModelLOG` with L-BFGS.

        Parameters
        ----------
        logr : np.ndarray
            Pairwise log-rigid observations, shape ``(6, K)``.
        session_list : list
            Session IDs to include (strings or objects with ``.id``).
        n_epochs : int
            Maximum number of L-BFGS epochs.
        cost : {'l1', 'l2'}
            Pairwise fitting loss.
        lr : float
            L-BFGS learning rate.
        dir_results : str
            Directory for :class:`~nicgiprep.callbacks.ModelCheckpoint`.
        max_iter : int, optional
            ``max_iter`` passed to L-BFGS. Default is 5.
        patience : int, optional
            Stop after this many epochs without ≥ 1e-4 improvement. Default
            is 3.
        device : str, optional
            PyTorch device string. Default is ``'cpu'``.
        verbose : bool, optional
            If ``True``, attach :class:`~nicgiprep.callbacks.PrinterCallback`.
            Default is ``True``.

        Returns
        -------
        np.ndarray
            Per-timepoint 4×4 rigid matrices, shape ``(4, 4, N)``.
        """
        if len(session_list) > 2:
            model = InstanceRigidModelLOG(session_list, cost=cost, device=device, reg_weight=0)
            optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr, max_iter=max_iter,
                                          line_search_fn='strong_wolfe')

            min_loss = 1000
            iter_break = 0
            log_dict = {}
            logr = torch.FloatTensor(logr)
            for epoch in range(n_epochs):

                def closure():
                    """L-BFGS closure: zero gradients, compute loss, backprop."""
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()

                    loss = model(logr, session_list)
                    loss.backward()

                    return loss

                optimizer.step(closure=closure)

                loss = model(logr, session_list)

                if loss < min_loss + 1e-4:
                    iter_break = 0
                    min_loss = loss.item()

                else:
                    iter_break += 1

                if iter_break > patience or loss.item() == 0.:
                    break

                log_dict['loss'] = loss.item()

            T = model.matrix

        else:
            logr = np.squeeze(logr.astype('float32'))
            model = InstanceRigidModelLOG(session_list, cost=cost, device=device, reg_weight=0)
            model.angle = nn.Parameter(torch.tensor(np.array([[-logr[0] / 2, logr[0] / 2],
                                                              [-logr[1] / 2, logr[1] / 2],
                                                              [-logr[2] / 2, logr[2] / 2]])).float(),
                                       requires_grad=False)

            model.translation = nn.Parameter(torch.tensor(np.array([[-logr[3] / 2, logr[3] / 2],
                                                                    [-logr[4] / 2, logr[4] / 2],
                                                                    [-logr[5] / 2, logr[5] / 2]])).float(),
                                             requires_grad=False)
            T = model.matrix

        return T


class USLRDeformable(LongitudinalProcessor):
    """Nonlinear longitudinal registration via BCH-approximated USLR.

    Estimates per-timepoint SVFs by solving a spanning-tree problem over
    pairwise SynthMorph deformation fields using L1 or L2 regression on
    a control-point grid.
    """

    @staticmethod
    def init_st2(session_list: list[str],
                 def_dir: str,
                 svf_shape: tuple[int],
                 factor: int=1,
                 mask_path: Optional[str]=None,
                 se: Optional[np.ndarray]=None,
                 penalty: float=1.) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Load pairwise SVFs and assemble the spanning-tree observation tensors.

        Parameters
        ----------
        session_list : list of str
            Ordered session IDs.
        def_dir : str
            Directory containing ``<ref>_to_<flo>.nii.gz`` SVF files.
        svf_shape : tuple of int
            Spatial shape of the SVF grid ``(X, Y, Z)``.
        factor : float, optional
            Multiplicative scale applied to loaded SVF values. Default is 1.
        mask_path : str, optional
            Path to a brain mask NIfTI resampled to SVF space. If ``None``,
            all voxels are included.
        se : array-like, optional
            Structuring element for binary dilation of the mask.
        penalty : float, optional
            Weight for the regularisation row added to the weight matrix.
            Default is 1.

        Returns
        -------
        phi : np.ndarray
            Pairwise SVF observations.
        obs_mask : np.ndarray
            Per-pair spatial masks.
        w : np.ndarray
            Weight matrix, shape ``(K+1, N)``.
        nk : int
            Number of pairs loaded.
        """
        sessions_dict = {t: it_t for it_t, t in enumerate(session_list)}

        N = len(session_list)
        K = int(N * (N - 1) / 2) + 1
        w = np.zeros((K, N), dtype='int')

        obs_mask = np.zeros(svf_shape + (K,))
        phi = np.zeros(svf_shape + (3, K,))

        nk = 0
        for tp_ref, tp_flo in itertools.combinations(session_list, 2):
            t0 = sessions_dict[tp_ref]
            t1 = sessions_dict[tp_flo]
            filename = str(tp_ref) + '_to_' + str(tp_flo)

            svf_proxy = nib.load(join(def_dir, filename + '.nii.gz'))
            svf_arr = np.asarray(svf_proxy.dataobj)

            # Masks
            if mask_path is not None:
                mask_proxy = nib.load(mask_path)
                mask_proxy = vol_resample_fast(svf_proxy, mask_proxy)
                mask_arr = np.array(mask_proxy.dataobj)

            else:
                mask_arr = np.ones(svf_shape)

            if se is not None:
                mask_arr = binary_dilation(mask_arr, se)

            phi[..., nk] = factor * svf_arr
            obs_mask[..., nk] = mask_arr

            w[nk, t0] = -1
            w[nk, t1] = 1
            nk += 1

        w[nk, :] = penalty
        nk += 1
        return phi, obs_mask, w, nk

    @staticmethod
    def st2_L2_global(phi: np.ndarray, w: np.ndarray, n_sess: int) -> np.ndarray:
        """Solve the ST² problem globally in L2 via the normal equations.

        Parameters
        ----------
        phi : np.ndarray
            Pairwise SVF observations, shape ``(*image_shape, 3, K)``.
        w : np.ndarray
            Weight matrix, shape ``(K+1, n_sess)``.
        n_sess : int
            Number of sessions.

        Returns
        -------
        np.ndarray
            Per-timepoint SVFs, shape ``(*image_shape, 3, n_sess)``.
        """
        precision = 1e-6
        lambda_control = np.linalg.inv((w.T @ w) + precision * np.eye(n_sess)) @ w.T
        Tres = lambda_control @ np.transpose(phi, [0, 1, 2, 4, 3])
        Tres = np.transpose(Tres, [0, 1, 2, 4, 3])

        return Tres

    @staticmethod
    def st2_L1(phi: np.ndarray,
               obs_mask: np.ndarray,
               w: np.ndarray,
               n_sess: int,
               chunk_id: Optional[int]=None,
               verbose: bool=True) -> np.ndarray:
        """Solve the ST2 problem voxel-wise in L1 via linear programming.

        Parameters
        ----------
        phi : np.ndarray
            Pairwise SVF observations, shape ``(*image_shape, 3, K)``.
        obs_mask : np.ndarray
            Per-pair spatial mask, shape ``(*image_shape, K)``.
        w : np.ndarray
            Weight matrix, shape ``(K+1, N)``.
        n_sess : int
            Number of sessions.
        chunk_id : int, optional
            Chunk identifier printed when processing a spatial sub-block.
        verbose : bool, optional
            If ``True``, print row progress. Default is ``True``.

        Returns
        -------
        np.ndarray
            Per-timepoint SVFs, shape ``(*image_shape, 3, n_sess)``.
        """
        if chunk_id is not None and verbose:
            print("Processing chunk " + str(chunk_id))

        image_shape = obs_mask.shape[:3]
        Tres = np.zeros(image_shape + (3, n_sess))

        for it_control_row in range(image_shape[0]):
            if np.mod(it_control_row, 10) == 0 and chunk_id is None and verbose:
                print('  * Row ' + str(it_control_row) + '/' + str(image_shape[0]))

            for it_control_col in range(image_shape[1]):
                for it_control_depth in range(image_shape[2]):
                    index_obs = np.where(obs_mask[it_control_row, it_control_col, it_control_depth, :] == 1)[0]

                    if index_obs.shape[0] > 0:
                        w_control = w[index_obs]
                        phi_control = phi[it_control_row, it_control_col, it_control_depth]
                        phi_control = phi_control[..., index_obs]
                        n_control = len(index_obs)

                        for it_dim in range(3):
                            # Set objective
                            c_lp = np.concatenate((np.ones((n_control,)), np.zeros((n_sess,))), axis=0)

                            # Set the inequality
                            A_lp = np.zeros((2 * n_control, n_control + n_sess))
                            A_lp[:n_control, :n_control] = -np.eye(n_control)
                            A_lp[:n_control, n_control:] = -w_control
                            A_lp[n_control:, :n_control] = -np.eye(n_control)
                            A_lp[n_control:, n_control:] = w_control

                            reg = np.reshape(phi_control[it_dim], (n_control,))
                            b_lp = np.concatenate((-reg, reg), axis=0)

                            result = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, bounds=(None, None), method='highs-ds')
                            Tres[it_control_row, it_control_col, it_control_depth, it_dim] = result.x[n_control:]

        return Tres

    @staticmethod
    def st2_L1_chunks(phi: np.ndarray,
                      obs_mask: np.ndarray,
                      w: np.ndarray,
                      n_sess: int,
                      num_chunks: int=2,
                      num_cores: int=4,
                      verbose: bool=True) -> np.ndarray:
        """Parallelise :meth:`st2_L1` by dividing the volume into chunks.

        Parameters
        ----------
        phi : np.ndarray
            Pairwise SVF observations, shape ``(*image_shape, 3, K)``.
        obs_mask : np.ndarray
            Spatial mask, shape ``(*image_shape, K)``.
        w : np.ndarray
            Weight matrix, shape ``(K+1, n_sess)``.
        N : int
            Number of sessions.
        num_chunks : int, optional
            Number of chunks per spatial dimension (total ``num_chunks³``
            jobs). Default is 2.
        num_cores : int, optional
            Parallel workers. ``1`` falls back to serial. Default is 4.
        verbose : bool, optional
            If ``True``, print row progress. Default is ``True``.

        Returns
        -------
        np.ndarray
            Per-timepoint SVFs, shape ``(*image_shape, 3, n_sess)``.
        """
        if num_cores == 1:
            Tres = USLRDeformable.st2_L1(phi, obs_mask, w, n_sess, verbose=verbose)

        else:
            chunk_list = []
            image_shape = obs_mask.shape[:3]
            chunk_size = [int(np.ceil(cs / num_chunks)) for cs in image_shape]
            for x in range(num_chunks):
                for y in range(num_chunks):
                    for z in range(num_chunks):
                        max_x = min((x + 1) * chunk_size[0], image_shape[0])
                        max_y = min((y + 1) * chunk_size[1], image_shape[1])
                        max_z = min((z + 1) * chunk_size[2], image_shape[2])
                        chunk_list += [[slice(x * chunk_size[0], max_x),
                                        slice(y * chunk_size[1], max_y),
                                        slice(z * chunk_size[2], max_z)]]

            results = Parallel(n_jobs=num_cores)(
                delayed(USLRDeformable.st2_L1)(
                    phi[chunk[0], chunk[1], chunk[2]], obs_mask[chunk[0], chunk[1], chunk[2]],
                    w, n_sess, chunk_id=it_chunk, verbose=verbose) for it_chunk, chunk in enumerate(chunk_list))

            Tres = np.zeros(phi.shape[:4] + (n_sess,))
            for it_chunk, chunk in enumerate(chunk_list):
                Tres[chunk[0], chunk[1], chunk[2]] = results[it_chunk]

        return Tres

    def _name(self):
        """Return the display name of this pipeline."""
        return 'Longitudinal:Deformable-Registration'

    def _build_processor(self):
        """Extend the base processor for nonlinear registration outputs."""
        super()._build_processor()
        self.tmp_dir = join(self.tmp_dir, 'long-lin-reg')
        create_dir(self.tmp_dir)
        self.pipeline_dir = 'nicgiprep-long'
        self.trajectory_ent = {'space': 'subject', 'task': 'linfit', 'scope': self.pipeline_dir, 'extension': 'nii.gz'}

    def _check_running_subject(self, subject: str, session_list: list[str], force_flag: bool=False) -> ProcessResult:
        """Determine the processing checkpoint for a subject.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str
            Available session IDs.
        force_flag : bool
            If ``True``, ignore existing outputs.

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}``. Exit codes:
            ``-1`` error, ``0`` run full pipeline, ``1`` skip,
            ``2`` SVF graph done, ``3`` eTIV done, ``4`` mean SVF done,
            ``5`` single timepoint.
        """
        # do not run if only 1 timepoint available
        if len(session_list) == 1:
            return ProcessResult(exit_code=5, message='[done] It has only 1 timepoint. Linking files. \n')

        # do not run if only 0 timepoint available
        elif len(session_list) == 0:
            return ProcessResult(exit_code=1, message='[done] It has 0 sessions available. Skipping.\n')

        # check if all timepoints are linearly registered
        elif any([self._get_data(**{'subject': subject, 'session': t, **self.im_long_ent}) is None for t in session_list]):
            return ProcessResult(exit_code=-1, message='[error] not all sessions are correctly registered to '
                                                       'subject space. Please check.\n')

        # check if the graph has been solved
        elif (self._get_data(**{'subject': subject, 'session': session_list, **self.svf_long_ent},
                             curr_len=len(session_list), verbose=False) is not None and not force_flag):

            filename_template = self.build_path({'suffix': 'T1w', 'subject': subject, **self.template_long_ent})
            if not exists(join(DIR_PIPELINES[self.pipeline_dir], filename_template)):
                return ProcessResult(exit_code=2, message='[partly done] graph already solved; '
                                                          'computing template and trajectories.\n')

            elif self._get_data(**{'subject': subject, 'suffix': 'jac', **self.trajectory_ent}, verbose=False) is None:
                return ProcessResult(exit_code=3, message='[partly done] graph, template done. '
                                                          'Computing mean trajectories.\n')

            else:
                return ProcessResult(exit_code=2, message='[done] subject already processed. Check the results in '
                                                          '[..]/' + self.pipeline_dir + '/sub-' + subject + '.\n')

        else:
            return ProcessResult(exit_code=0, message='Subject needs to be processed')


    def _init_graph(self, subject: str, session_list: list[str], def_dir: str, force_flag: bool=False) -> None:
        """Register all pairs of timepoints with SynthMorph and save the SVFs.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str
            Session IDs to include
        def_dir : str
            Directory where ``<ref>_to_<flo>.nii.gz`` SVF files are written.
        force_flag : bool, optional
            If ``True``, recompute even when files exist. Default is
            ``False``.
        """
        svf_v2r = np.load(self._get_data(**{'subject': subject, **self.svf_v2r_ent}).path)
        for sess_ref, sess_flo in itertools.permutations(session_list, 2):
            output_filepath = join(def_dir, str(sess_ref) + '_to_' + str(sess_flo) + '.nii.gz')
            if exists(output_filepath) and not force_flag:
                continue

            # read image and mask
            imref_file = self._get_data(**{'subject': subject, 'session': sess_ref, **self.im_long_ent})
            imflo_file = self._get_data(**{'subject': subject, 'session': sess_flo, **self.im_long_ent})

            if imref_file is None or imflo_file is None:
                continue

            fw_svf = synthmorph_register(imref_file, imflo_file)
            if fw_svf is None:
                return ProcessResult(exit_code=-1, message="[error] deformable registration has failed.")
            else:
                save_volume(fw_svf, svf_v2r, output_filepath)

    def _solve_graph(self,
                     session_list: list[str],
                     def_dir: str,
                     cost: Literal['bch-l1', 'bch-l2']='bch-l2') -> dict:

        """Solve the deformable spanning-tree problem with BCH approximation.

        Parameters
        ----------
        session_list : list of str
            Session IDs to include,
        def_dir : str
            Directory with pairwise SVF NIfTI files.
        cost : {'bch-l1', 'bch-l2'}
            Optimisation strategy: BCH variants use the additive SVF
            approximation; plain L1/L2 use linear programming.

        Returns
        -------
        dict
            ``{session_id: np.ndarray}`` per-timepoint SVFs,
            shape ``(*svf_shape, 3)``.
        """
        R, M, W, NK = USLRDeformable.init_st2(session_list, def_dir, self.svf_shape, se=None)

        if cost == 'bch-l2':
            T_latent = USLRDeformable.st2_L2_global(R, W, len(session_list))
            T_latent = {t: T_latent[..., it_t] for it_t, t in enumerate(session_list)}

        else:
            T_latent = USLRDeformable.st2_L1_chunks(R, M, W, len(session_list), num_cores=1)
            T_latent = {t: T_latent[..., it_t] for it_t, t in enumerate(session_list)}

        return T_latent

    def _compute_template(self, subject: str, sess_df: pd.DataFrame) -> ProcessResult|None:
        """Build the nonlinear template image, segmentation, mask, and eTIV.

        Warps each timepoint to the linear template space using the estimated
        SVFs, takes the median image and majority-vote segmentation, and
        saves the results to the ``uslr`` derivative.

        Parameters
        ----------
        subject : str
            Subject ID.
        sess_df : pd.DataFrame
            DataFrame with session_id as index and orig_t1w and orig_seg as columns
        """
        sss_kwargs = self.template_long_ent.copy()
        sss_kwargs['subject'] = 'subject'
        sss_kwargs['suffix'] = 'empty'
        sss_kwargs['datatype'] = 'utils'
        sss_file = self._get_data(**{'subject': subject, **sss_kwargs})

        if sss_file is None:
            return ProcessResult(exit_code=-1, message='[error] Subject-space has not been created. Please check.\n')

        sss_proxy = nib.load(sss_file.path)

        # build path template: image, mask, seg
        image_filename = self.build_path({'suffix': 'T1w', 'subject': subject, **self.template_long_ent})
        seg_filename = self.build_path({'suffix': 'T1wdseg', 'subject': subject, **self.template_long_ent})

        # compute template: image, mask, seg
        image_list = []
        seg_list = []
        for sess_id, sess_files in sess_df.iterrows():
            im_file = sess_files['orig_t1w']
            seg_file = sess_files['orig_seg']
            aff_file = self._get_data(**{'subject': subject, 'session': sess_id, **self.aff_long_ent})
            svf_file = self._get_data(**{'subject': subject, 'session': sess_id, **self.svf_long_ent})

            if svf_file is None or aff_file is None:
                return ProcessResult(exit_code=-1,
                                     message='[error] Some intermediate steps have failed. Please check.\n')

            im_proxy = nib.load(im_file.path)
            seg_proxy = nib.load(seg_file.path)
            aff_arr = np.load(aff_file.path)
            svf_proxy = nib.load(svf_file.path)
            flow_arr = integrate_svf(np.array(svf_proxy.dataobj), self.net_shape, scaling_factor=2, int_steps=7)
            flow_proxy = nib.Nifti1Image(flow_arr, affine=sss_proxy.affine)

            # Image
            im_arr = np.array(im_proxy.dataobj)
            im_arr = gaussian_antialiasing(im_arr, im_proxy.affine, [1, 1, 1])
            im_proxy = nib.Nifti1Image(im_arr, np.linalg.inv(aff_arr) @ im_proxy.affine)
            im_proxy = vol_resample_fast(sss_proxy, im_proxy, proxyflow=flow_proxy)
            im_proxy.uncache()

            seg_arr = np.array(seg_proxy.dataobj)
            onehot_arr = one_hot_encoding(seg_arr, categories=SYNTHSEG_APARC_LUT).astype('float')
            onehot_proxy = nib.Nifti1Image(onehot_arr, np.linalg.inv(aff_arr) @ seg_proxy.affine)
            onehot_proxy = vol_resample_fast(sss_proxy, onehot_proxy, proxyflow=flow_proxy)
            onehot_proxy.uncache()

            image_list.append(im_proxy)
            seg_list.append(onehot_proxy)

        # save image (median and std), mask (and etiv) and seg.
        image_list_arr = np.stack([np.array(im_proxy.dataobj) for im_proxy in image_list], axis=0)
        im_template_arr = np.median(image_list_arr, axis=0)
        del image_list

        seg_template_arr = np.zeros(im_template_arr.shape + (len(SYNTHSEG_APARC_LUT),))
        for seg_proxy in seg_list:
            seg_template_arr += np.array(seg_proxy.dataobj)

        del seg_list
        seg_template_arr = np.argmax(seg_template_arr, axis=-1)
        seg_template_arr = self._undo_one_hot(seg_template_arr)

        save_volume(im_template_arr, sss_proxy.affine, join(DIR_PIPELINES[self.pipeline_dir], image_filename))
        save_volume(seg_template_arr, sss_proxy.affine, join(DIR_PIPELINES[self.pipeline_dir], seg_filename))

    def _compute_mean_trajectories(self, subject: str, session_list: list[str]) -> ProcessResult|None:
        """Fit a linear trajectory through per-timepoint SVFs and save statistics.

        Runs ordinary least squares over time to decompose per-timepoint
        SVFs into an intercept and a slope (rate-of-change SVF). Also
        integrates the slope SVF and saves the Jacobian determinant map.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str
            Session IDs to include
        """
        linreg = LinearRegression()
        time_list = self._get_session_time(subject, session_list)

        svf_filename = self.build_path({'subject': subject, 'suffix': 'svf', **self.trajectory_ent})
        flow_filename = self.build_path({'subject': subject, 'suffix': 'def', **self.trajectory_ent})
        jac_filename = self.build_path({'subject': subject, 'suffix': 'jac', **self.trajectory_ent})

        net_v2r_file = self._get_data(**{'subject': subject, **self.net_v2r_ent})
        svf_v2r_file = self._get_data(**{'subject': subject, **self.svf_v2r_ent})
        if svf_v2r_file is None or net_v2r_file is None:
            return ProcessResult(exit_code=-1, message='[error] Some error occurred in the rigid registration step. '
                                                       'Please check.\n')

        net_v2r = np.load(net_v2r_file.path)
        svf_v2r = np.load(svf_v2r_file.path)

        svf_list = []
        features_list = []
        for sess_id in session_list:
            svf_file = self._get_data(**{'subject': subject, 'session': sess_id, **self.svf_long_ent})
            if svf_file is None:
                return ProcessResult(exit_code=-1,
                                     message='[error] Subject SVFs have not been computed. Please check.\n')

            svf_proxy = nib.load(svf_file.path)
            svf_list.append(np.array(svf_proxy.dataobj).reshape(-1))

            age = float(time_list[sess_id])
            features_list.append([age])

        X = np.array(features_list)
        Y = np.stack(svf_list, axis=0)
        linreg.fit(X, Y)

        coef_list = [linreg.coef_[:, it_f].reshape(self.svf_shape + (3,)) for it_f in range(len(features_list[0]))]
        intercept_list = [linreg.intercept_.reshape(self.svf_shape + (3,))]
        results_vol = np.stack(intercept_list + coef_list, axis=-1)
        save_volume(results_vol, svf_v2r, join(DIR_PIPELINES[self.pipeline_dir], svf_filename))

        svf = results_vol[..., 1]
        if max(time_list.values()) - min(time_list.values()) > 30:
            svf = svf * 365.25
        flow = integrate_svf(svf, self.net_shape, scaling_factor=2, int_steps=7)
        save_volume(flow, net_v2r, join(DIR_PIPELINES[self.pipeline_dir], flow_filename))

        jac = compute_jacobian(flow)
        save_volume(jac, net_v2r, join(DIR_PIPELINES[self.pipeline_dir], jac_filename))


    def process_subject(self,
                        subject: str,
                        cost: Literal['bch-l1', 'bch-l2']='bch-l2',
                        force_flag: bool=False,
                        **kwargs) -> ProcessResult:
        """Run the full nonlinear USLR pipeline for one subject.

        Orchestrates: SVF graph initialisation (pairwise SynthMorph),
        spanning-tree solve, template construction, mean SVF fitting,
        and optional MNI registration. Skips completed steps.

        Parameters
        ----------
        subject : str
            Subject ID.
        cost : {'bch-l1', 'bch-l2'}, optional
            Optimisation strategy for the spanning-tree solve. Default is ``'bch-l1'``.
        force_flag : bool, optional
            If ``True``, rerun all steps. Default is ``False``.
        **kwargs
            Forwarded to :meth:`_solve_graph`.
        """
        exit_dict = ProcessResult(exit_code=0, message='success')
        try:
            assert cost in ['bch-l1', 'bch-l2']

            sess_df = self._get_sessions_file(subject)
            if not isinstance(sess_df, pd.DataFrame):
                if kwargs.get('verbose', False): print(sess_df['message'])
                return sess_df

            session_list = sess_df['session_id']
            sess_df.set_index('session_id', drop=False, inplace=True)

            checkpoint = self._check_running_subject(subject, session_list, force_flag)
            if kwargs.get('verbose', False):
                print('* Subject: ' + subject)

            if checkpoint['exit_code'] == -1 or checkpoint['exit_code'] == 1:
                if kwargs.get('verbose', False): print(checkpoint['message'])
                return checkpoint

            if checkpoint['exit_code'] in [5]:
                if kwargs.get('verbose', False): print(checkpoint['message'])
                return checkpoint

            def_dir = join(self.tmp_dir, 'sub-' + subject)
            create_dir(def_dir)

            if checkpoint['exit_code'] in [0]:
                # compute svf v2r
                svf_v2r_file = self._get_data(subject=subject, **self.svf_v2r_ent)
                if svf_v2r_file is None:
                    return ProcessResult(exit_code=-1, message="[error] please, something went wrong in the rigid "
                                                               "registration step. Please check.")
                else:
                    svf_v2r = np.load(svf_v2r_file.path)

                # build the entire graph
                self._init_graph(subject, session_list, def_dir, force_flag)
                self._update_subject_layout(subject)

                # solve spanning tree
                T_latent = self._solve_graph(session_list, def_dir, cost)
                for sess_id in sess_df.index:
                    filename = self.build_path({'subject': subject, 'session': sess_id, **self.svf_long_ent})
                    filepath = join(DIR_PIPELINES['nicgiprep-long'], filename)
                    create_dir(dirname(filepath))
                    save_volume(T_latent[sess_id].astype('float32'), svf_v2r, path=filepath)

                self._update_subject_layout(subject)

            if checkpoint['exit_code'] in [0, 2]:
                # compute template
                self._compute_template(subject, sess_df)
                self._update_subject_layout(subject)

            if checkpoint['exit_code'] in [0, 2, 3]:
                # compute mean SVF
                self._compute_mean_trajectories(subject, session_list)
                self._update_subject_layout(subject)

            return exit_dict

        except Exception as e:
            if kwargs.get('verbose', False): print(traceback.format_exc())
            return ProcessResult(exit_code=-1, message=f'[error] subject {subject} failed: {e}')


class LongitudinalRegistration(LongitudinalProcessor):

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
        super().__init__(bids_loader=bids_loader, subject_list=subject_list, **kwargs)

        self.linear_reg = USLRLinear(bids_loader=bids_loader, subject_list=subject_list, **kwargs)
        self.nonlinear_reg = USLRDeformable(bids_loader=bids_loader, subject_list=subject_list, **kwargs)


    def _build_processor(self):
        """Extend the base processor with longitudinal registration tmp and pipeline directories."""
        super()._build_processor()

        self.tmp_dir = join(self.tmp_dir, 'long-reg')
        create_dir(self.tmp_dir)
        self.pipeline_dir = 'nicgiprep-long'



    def _check_running_subject(self,
                               subject: str,
                               session_list: list[str],
                               force_flag: bool = False,
                               register_MNI: bool = False) -> dict:

        """Determine the processing checkpoint for a subject.

        Parameters
        ----------
        subject : str
            Subject ID of the processing subject.
        session_list : list of str
            Session IDs to include in the processing.
        force_flag : bool, optional
            If ``True``, ignore existing outputs and rerun.
        register_MNI : bool, optional
            Whether MNI registration is expected. Default is ``False``.

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}``.  Exit codes:
            ``-1`` error, ``0`` run full pipeline, ``1`` skip,
            ``2`` graph done, ``3`` template done, ``4`` eTIV done,
            ``5`` single timepoint.
        """
        # do not run if only 1 timepoint available
        if len(session_list) == 1:
            if register_MNI:
                return {'exit_code': 5,
                        'message': '[partly done] It has only 1 timepoint. Linking files and registering to MNI \n'}
            else:
                return {'exit_code': 5,
                        'message': '[done] It has only 1 timepoint. Linking files. \n'}

        # do not run if only 0 timepoint available
        elif len(session_list) == 0:
            return {'exit_code': 1, 'message': '[done] It has 0 sessions available. Skipping.\n'}

        # do not run if some timepoint is not properly segmented
        elif any([self._get_data(**{'subject': subject, 'session': t, **self.seg_entities}) is None for t in
                  session_list]):
            return {'exit_code': -1, 'message': '[error] not all sessions are correctly segmented. Please check.\n'}

        # do not run if it has already been processed
        elif self._get_data(**{'subject': subject, **self.aff_long_ent}, curr_len=len(session_list), verbose=False) is not None and not force_flag:
            filename_sss = self.build_path({'subject': subject, **self.template_long_entities})

            if not exists(join(DIR_PIPELINES[self.pipeline_dir], filename_sss)):
                return {'exit_code': 2,
                        'message': '[partly done] graph is already computed; template and etiv missing.\n'}
            elif self._get_data(**{**self.im_long_ent, 'subject': subject, 'suffix': 'T1wdseg'}, curr_len=len(session_list), verbose=False) is None:
                return {'exit_code': 2,
                        'message': '[partly done] graph is already computed; template and etiv missing.\n'}
            elif not exists(join(DIR_PIPELINES[self.pipeline_dir], 'sub-' + subject, 'sub-' + subject + '_T1wetiv.npy')):
                return {'exit_code': 3,
                        'message': '[partly done] graph and template are already computed; subject etiv missing.\n'}
            elif self._get_data(**{'subject': subject, 'space': 'MNI', 'suffix': 'T1wdseg'},
                                curr_len=len(session_list), verbose=False) is None and register_MNI is True:
                return {'exit_code': 4,
                        'message': '[partly done] graph, template and etiv done; MNI registration is missing.\n'}
            else:
                return {'exit_code': 1,
                        'message': '[done] subject already processed. '
                                   'Check the results in [..]/uslr-lin/sub-' + subject + '.\n'}

        # do not run if more segmentations than sessions are found
        elif self._get_data(**{'subject': subject, **self.seg_entities}, curr_len=len(session_list)) is None:
            return {'exit_code': -1,
                    'message': '[error] not all sessions are segmented. Please, run preprocess/synthseg.py first.\n'}

        else:
            return {'exit_code': 0, 'message': ''}

    def _register_to_MNI(self, subject: str, sess_df: pd.DataFrame):
        """Register the nonlinear template to MNI via centroid affine.

        Saves the affine, SVF, and Jacobian determinant in MNI space.

        Parameters
        ----------
        subject : str
            Subject ID.
        sess_df : pd.DataFrame
            DataFrame with session_id as index and orig_t1w and orig_seg as columns
        """
        mni_ent = {'subject': subject, 'space': 'MNI'}
        aff_fname = self.build_path({'suffix': 'aff', 'extension': 'npy', 'datatype': 'utils', 'desc': 'tosubject', **mni_ent})

        template_im = self._get_data(**{'suffix': 'T1w', 'subject': subject, **self.template_long_ent})
        template_seg = self._get_data(**{'suffix': 'T1wdseg', 'subject': subject, **self.template_long_ent})
        if template_seg is None or template_im is None:
            return ProcessResult(exit_code=-1, message='[error] subject ' + subject + ' does not have a template.')

        centroid_ref, ok_ref = compute_centroids_ras(MNI_TEMPLATE_SEG, labels_registration)
        centroid_flo, ok_flo = compute_centroids_ras(template_seg.path, labels_registration)

        aff_MNI = getM(centroid_ref[:, ok_ref > 0], centroid_flo[:, ok_ref > 0], use_L1=False)
        np.save(join(DIR_PIPELINES[self.pipeline_dir], aff_fname), aff_MNI)

        mni_proxy = nib.load(MNI_TEMPLATE)
        for sess_id, sess_files in sess_df.iterrows():
            im_file = sess_files['orig_t1w']
            extra_kwargs = {'subject': subject, 'session': sess_id}
            aff_file = self._get_data(**{**extra_kwargs, **self.aff_long_ent})
            im_fname = self.build_path({'session': sess_id, 'suffix': 'T1w', 'extension': 'nii.gz', **mni_ent})

            if aff_file is None:
                return ProcessResult(exit_code=-1, message='[error] Something went wrong in the rigid registration '
                                                           'step for' + str(extra_kwargs) + '.\n')

            aff = np.load(aff_file)
            if np.sum(np.isnan(aff)) > 0:
                return ProcessResult(exit_code=-1, message='[error] Something went wrong in the rigid registration '
                                                           'step for' + str(extra_kwargs) + '.\n')

            im_proxy = nib.load(im_file.path)
            voxsize = np.sqrt(np.sum(im_proxy.affine * im_proxy.affine, axis=0))[:-1]
            voxsize_new = np.sqrt(np.sum(mni_proxy.affine * mni_proxy.affine, axis=0))[:-1]
            factor = voxsize / voxsize_new
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling

            im_array = np.array(im_proxy.dataobj)
            im_array = gaussian_filter(im_array, sigmas)
            im_proxy = nib.Nifti1Image(im_array, np.linalg.inv(aff_MNI) @ np.linalg.inv(aff) @ im_proxy.affine)
            im_proxy = vol_resample_fast(mni_proxy, im_proxy)

            nib.save(im_proxy, join(DIR_PIPELINES[self.pipeline_dir], im_fname))

        return ProcessResult(exit_code=0, message='succeed')

    def process_subject(self, subject: str, force_flag: bool=False, **kwargs) -> ProcessResult:

        # build sessions file with the filename used for each session (T1w)
        sess_df = self._get_sessions_file(subject)
        if not isinstance(sess_df, pd.DataFrame):
            if kwargs.get('verbose', False): print(sess_df['message'])
            return sess_df

        # call linear --> save etiv and T1w in subject space
        lin_res = self.linear_reg.process_subject(subject=subject, force_flag=force_flag, **kwargs)
        self._update_subject_layout(subject)

        # call nonlinear --> save nonlinear templates (T1w, T1wdseg) and SVF and JAC.
        nonlin_res = self.nonlinear_reg.process_subject(subject=subject, force_flag=force_flag, **kwargs)
        self._update_subject_layout(subject)

        # deform nonlinear template to MNI
        self._register_to_MNI(subject, sess_df)

        return ProcessResult(exit_code=0, message='success')


