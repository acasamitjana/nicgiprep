"""
Processing pipeline classes for longitudinal neuroimaging data in BIDS format.

Provides base and specialised pipeline classes that wrap PyBIDS layout queries,
coordinate volumetric resampling, label fusion, and longitudinal volume tracking.
"""

import pdb
import traceback
from typing import Optional, Literal
from os.path import isfile, join, dirname, basename, exists
from os import makedirs
from joblib import delayed, Parallel
import itertools

from bids.layout import BIDSLayout
from sympy.tensor.array import ndim_array
from torch import nn
from skimage.morphology import ball, binary_dilation
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import numpy as np
import nibabel as nib
import pandas as pd
import surfa as sf


from setup import *
from nicgiprep.pipelines.base import Processor
from nicgiprep.models import InstanceRigidModelLOG, ST2Nonlinear
from nicgiprep.callbacks import *
from nicgiprep.utils.preprocessing_utils import *
from nicgiprep.utils.label_utils import SYNTHSEG_APARC_LUT
from nicgiprep.utils.io_utils import create_dir, save_volume, remove_dir, ProcessResult
from nicgiprep.utils.synthmorph_utils import synthmorph_register, integrate_svf, compose_transforms
from nicgiprep.utils.def_utils import vol_resample_fast, network_space, create_empty_template, compute_jacobian, getM
from nicgiprep.utils.fn_utils import one_hot_encoding, rescale_voxel_size, compute_centroids_ras, gaussian_antialiasing
from nicgiprep.utils.synthmorph_utils import warp



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
        - ``mask_long_entities`` — entities for brain masks in USLR space.
        - ``svf_long_entities`` — entities for nonlinear SVF graph files.
        - ``template_long_entities`` — entities for the linear template.
        - ``net_shape`` / ``svf_shape`` — default network and SVF spatial shapes.
        - ``net_v2r_entities`` / ``svf_v2r_entities`` — v2r affine file entities.
        """
        super()._build_processor()
        self.long_entities = {'space': 'subject', 'acquisition': '1', 'extension': 'nii.gz'}
        self.aff_long_ent = {'desc': 'raw2temp', 'suffix': 'aff', 'extension': '.npy'}
        self.im_long_ent = {'suffix': 'T1w', **self.long_entities}
        self.mask_long_entities = {'suffix': 'T1wmask', **self.long_entities}
        self.svf_long_entities = {'suffix': 'svf', 'extension': 'nii.gz', 'space': 'uslr', 'scope': 'nonlin'}
        self.template_long_entities = {'desc': 'template', 'suffix': 'T1w', **self.long_entities}

        self.net_shape = (192, 192, 192)
        self.svf_shape = (96, 96, 96)

        self.net_v2r_entities = {'desc': 'template', 'suffix': 'v2r', 'extension': '.npy', 'space': 'subject'}
        self.svf_v2r_entities = {'desc': 'svf', 'suffix': 'v2r', 'extension': '.npy', 'space': 'subject'}

        self.tmp_dir = join(self.tmp_dir, 'long')


    def _name(self) -> str:
        """Return the display name of this pipeline."""
        return 'LongitudinalProcessor'


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

    def _check_running_subject(self, subject: str,
                               session_list: list[str],
                               force_flag: bool = False) -> ProcessResult:

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
            ``2`` graph done, ``3`` template done, ``4`` eTIV done,
            ``5`` single timepoint.
        """
        # do not run if only 1 timepoint available
        if len(session_list) == 1:
            return ProcessResult(exit_code=5, message='[done] It has only 1 timepoint. Linking files. \n')

        # do not run if only 0 timepoint available
        elif len(session_list) == 0:
            return ProcessResult(exit_code=1, message='[done] It has 0 sessions available. Skipping.\n')

        # do not run if some timepoint is not properly segmented
        elif any([self._get_data(**{'subject': subject, 'session': t, **self.seg_entities})
                  is None for t in session_list]):
            return ProcessResult(exit_code=-1, message='[error] not all sessions are correctly segmented. Please check.\n')

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

    def _get_centroids(self, subject: str, session_list: list[str]):
        """Compute RAS centroids (mm) for each ROI segmented on all the available sessions .
        Each segmentation provides N_label ROIs.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str
            Session IDs to include

        Returns
        -------
        centroid_dict : dict
            ``{session_id: np.ndarray}`` of shape ``(3, N_labels)``.
        ok : dict
            ``{session_id: np.ndarray}`` binary flags per label.
        """
        centroid_dict = {}
        ok = {}
        for sess_id in session_list:
            seg_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': sess_id})
            centroid_dict[sess_id], ok[sess_id] = compute_centroids_ras(seg_file.path, labels_registration)

        return centroid_dict, ok

    def _compute_cog(self, subject: str, session_list: list[str]) -> dict:
        """Compute and save a centring-to-COG transform for each session.

        The centre-of-gravity (COG) is computed from the non-zero voxels of
        each segmentation and saved as a 4×4 translation matrix in RAS mm.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str
            Session IDs to include.
        """
        T_cog_d = {}
        for sess_id in session_list:
            seg_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': sess_id})

            seg_proxy = nib.load(seg_file.path)
            data = np.array(seg_proxy.dataobj)
            aux = np.where(data>0)
            i, j, k = np.median(aux[0]), np.median(aux[1]), np.median(aux[2])
            ras_cog = seg_proxy.affine @ np.array([i, j, k, 1])
            T_cog = np.eye(4)
            T_cog[:3, -1] = -ras_cog[:3]
            T_cog_d[sess_id] = T_cog

        return T_cog

    def _init_graph(self, subject: str, session_list: list[str], def_dir: str, force_flag: bool=False) -> dict:
        """Compute pairwise rigid affines between all timepoints via centroid SVD.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str
            Session IDs to include.
        def_dir : str
            Directory where pairwise ``.npy`` affine files are written.
        force_flag : bool
            If ``True``, recompute even when files exist.
        """
        # compute centroids
        t_cog_d = self._compute_cog(subject, session_list)
        centroids_dict, ok_dict = self._get_centroids(subject, session_list)

        for sess_id in session_list:
            t_cog = t_cog_d[sess_id]
            centroids_dict[sess_id] = t_cog @ np.concatenate([centroids_dict[sess_id], np.ones((1, centroids_dict[sess_id].shape[1]))])
            centroids_dict[sess_id] = centroids_dict[sess_id][:3]

        # pairwise registration
        for sess_ref, sess_flo in itertools.combinations(session_list, 2):
            output_filepath = join(def_dir, str(sess_ref) + '_to_' + str(sess_flo) + '.npy')
            ok_cent = (ok_dict[sess_ref] == 1) & (ok_dict[sess_flo] == 1)
            self._register_timepoints(centroids_dict[sess_ref], centroids_dict[sess_flo], output_filepath,
                                      ok_centr=ok_cent, force_flag=force_flag)

        return t_cog_d

    def _solve_graph(self, subject: str, session_list: list[str], def_dir: str, t_cog_d: dict, **kwargs) -> None:
        """Solve the rigid spanning-tree problem and save per-timepoint affines.

        Reads pairwise log-rigid observations, fits the USLR model via
        L-BFGS, and writes one ``.npy`` affine per timepoint.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str
            Session IDs to include.
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
        log_r = USLRLinear.init_st2_lineal(session_list, def_dir)
        t_res = USLRLinear.st2_lineal_pytorch(log_r, session_list, verbose=False, **kwargs)

        if np.sum(np.isnan(t_res)) > 0:
            return {'exit_code': -1, 'message': '[error] Something went wrong in the rigid registration step.\n'}

        for it_sess_id, sess_id in enumerate(session_list):

            extra_kwargs = {'session': sess_id, 'subject': subject}
            filename = self.build_path({**extra_kwargs, **self.aff_long_ent})

            affine_matrix = t_res[..., it_sess_id]
            T_cog = t_cog_d[sess_id]

            output_filepath = join(DIR_PIPELINES[self.pipeline_dir], filename)
            create_dir(dirname(output_filepath))

            np.save(output_filepath, np.linalg.inv(T_cog) @ affine_matrix)

        return None

    # def _create_subject_space(self, subject: str, session_list: list[str]):
    #     """Build a 1 mm isotropic network-space template for the subject.
    #
    #     Computes an average bounding box from all timepoints' brain masks,
    #     defines a network space (LIA, 1 mm, 192³), resamples each session's
    #     image and segmentation there, takes the median image and majority-vote
    #     segmentation, and saves everything to ``uslr-lin``.
    #
    #     Parameters
    #     ----------
    #     subject : str
    #         Subject ID.
    #     timepoints : list of str
    #         Session IDs to include
    #
    #     Returns
    #     -------
    #     dict
    #         Checkpoint dict with ``exit_code=3``.
    #     """
    #     # load segs, binarize, dilate and crop with 5 voxels per side.
    #     aff = {}
    #     masks = {}
    #     masks_dilated = {}
    #     orig_v2r = {}
    #     for sess_id in session_list:
    #         # filename = self.build_path({'session': sess_id, 'subject': subject, **self.aff_long_ent})
    #         aff_file = self._get_data(**{'session': sess_id, 'subject': subject, **self.aff_long_ent})
    #         seg_file = self._get_data(**{'session': sess_id, 'subject': subject, **self.seg_entities})
    #         if aff_file is not None and seg_file is not None:
    #             m = np.load(aff_file)
    #             if np.sum(np.isnan(m)) > 0:
    #                 return {'exit_code': -1, 'message': '[error] Something went wrong in the rigid registration step.\n'}
    #
    #             aff[tp] = m
    #
    #             proxyseg = nib.load(seg_file)
    #             orig_v2r[tp] = proxyseg.affine
    #
    #             seg = np.array(proxyseg.dataobj)
    #             mask = (seg > 0) & (seg != 24)
    #             masks[tp] = nib.Nifti1Image(mask.astype('float'), np.linalg.inv(m) @ proxyseg.affine)
    #
    #             mask_dilated = binary_dilation(mask, ball(3)).astype('float')
    #             masks_dilated[tp] = nib.Nifti1Image(mask_dilated, np.linalg.inv(m) @ proxyseg.affine)
    #
    #     # create subject space
    #     rasMosaic, template_vox2ras0, template_size = create_empty_template(list(masks_dilated.values()))
    #     save_nii(np.zeros(template_size), template_vox2ras0, join(self.tmp_dir, subject + '_template.nii.gz'))
    #
    #     # move subject space to network space
    #     template = sf.load_volume(join(self.tmp_dir, subject + '_template.nii.gz'))
    #     net2vox, vox2net, net_v2r = network_space(template, shape=self.net_shape, center=template)
    #     proxytemplate = nib.Nifti1Image(np.zeros(self.net_shape), net_v2r)
    #
    #     filename_ssspace = self.build_path({'subject': subject, **self.template_long_entities})
    #     filename_ssseg = self.build_path({'subject': subject, **self.template_long_entities, 'suffix': 'T1wdseg'})
    #     filename_ssmask = self.build_path({'subject': subject, **self.template_long_entities, 'suffix': 'T1wmask'})
    #     filename_t_v2r = self.build_path({'subject': subject, **self.net_v2r_entities})
    #
    #     create_dir(dirname(join(DIR_PIPELINES[self.pipeline_dir], filename_t_v2r)))
    #     np.save(join(DIR_PIPELINES[self.pipeline_dir], filename_t_v2r), net_v2r)
    #     os.remove(join(self.tmp_dir, subject + '_template.nii.gz'))
    #
    #     # resample each timepoint to network space (images and dilated masks)
    #     image_list = []
    #     seg_list = []
    #     for tp in timepoints:
    #         proxymask = vol_resample_fast(proxytemplate, masks[tp])
    #
    #         image_file = self._get_data(**{'subject': subject, 'session': tp, **self.bf_entities})
    #         if image_file is None:
    #             continue
    #
    #         proxyraw = nib.load(image_file.path)
    #         pixdim = np.sqrt(np.sum(proxyraw.affine * proxyraw.affine, axis=0))[:-1]
    #         new_vox_size = np.array([1, 1, 1])
    #         factor = pixdim / new_vox_size
    #         sigmas = 0.25 / factor
    #         sigmas[factor > 1] = 0  # don't blur if upsampling
    #
    #         im_array = np.array(proxyraw.dataobj)
    #         im_array = gaussian_filter(im_array, sigmas)
    #         proxyraw = nib.Nifti1Image(im_array, np.linalg.inv(aff[tp]) @ proxyraw.affine)
    #         proxyraw = vol_resample_fast(proxytemplate, proxyraw)
    #         image_list.append(proxyraw)
    #
    #         seg_file = self._get_data(**{'subject': subject, 'session': tp, **self.seg_entities})
    #         proxyseg = nib.load(seg_file.path)
    #         arrayseg = np.array(proxyseg.dataobj)
    #         proxyseg = nib.Nifti1Image(arrayseg, np.linalg.inv(aff[tp]) @ proxyseg.affine)
    #         proxyseg = vol_resample_fast(proxytemplate, proxyseg, mode='nearest')
    #         seg_list.append(proxyseg)
    #         # arrayseg = np.array(proxyseg.dataobj)
    #         # arrayonehot = one_hot_encoding(arrayseg, categories=SYNTHSEG_APARC_LUT).astype('float')
    #         # proxyonehot = nib.Nifti1Image(arrayonehot, np.linalg.inv(aff[tp]) @ proxyseg.affine)
    #         # proxyonehot = vol_resample_fast(proxytemplate, proxyonehot)
    #         # proxyonehot.uncache()
    #         # seg_list.append(proxyonehot)
    #
    #         # saving
    #         extra_kwargs = {'subject': subject, 'session': tp}
    #         filename_mask = self.build_path({**extra_kwargs, **self.mask_long_entities})
    #         filename_im = self.build_path({**extra_kwargs, **self.im_long_entities})
    #         filename_seg = self.build_path({**extra_kwargs, **self.im_long_entities, 'suffix': 'T1wdseg'})
    #
    #         nib.save(proxymask, join(DIR_PIPELINES[self.pipeline_dir], filename_mask))
    #         nib.save(proxyraw, join(DIR_PIPELINES[self.pipeline_dir], filename_im))
    #         nib.save(proxyseg, join(DIR_PIPELINES[self.pipeline_dir], filename_seg))
    #
    #     temp_array = np.stack([np.array(x.dataobj) for x in image_list], axis=0)
    #     temp_array = np.median(temp_array, axis=0)
    #     save_nii(temp_array, net_v2r, join(DIR_PIPELINES[self.pipeline_dir], filename_ssspace))
    #
    #     template_seg = np.zeros(proxytemplate.shape + (len(SYNTHSEG_APARC_LUT),))
    #     for proxyseg in seg_list:
    #         template_seg += one_hot_encoding(np.array(proxyseg.dataobj), categories=SYNTHSEG_APARC_LUT).astype('float')
    #         # template_seg += np.array(proxyseg.dataobj)
    #
    #     template_seg = np.argmax(template_seg, axis=-1)
    #     template_seg = self._undo_one_hot(template_seg)
    #     template_mask = (template_seg > 0) & (template_seg != 24)
    #
    #     save_nii(template_seg, net_v2r, join(DIR_PIPELINES[self.pipeline_dir], filename_ssseg))
    #     save_nii(template_mask, net_v2r, join(DIR_PIPELINES[self.pipeline_dir], filename_ssmask))
    #
    #     return {'exit_code': 3, 'message': '[partly done] graph and template are already computed; subject etiv missing.\n'}

    def _compute_etiv(self, subject: str, session_list: list[str]) -> ProcessResult:
        """Estimate and save the total intra-cranial volume (eTIV) for a subject.

        Averages binary brain masks across timepoints in the network space and
        saves the resulting voxel count.

        Parameters
        ----------
        subject : str
            Subject ID.
        timepoints : list of str
            Session IDs.
        """

        masks, masks_dilated = [], []
        template_mask = np.zeros(self.net_shape + (len(SYNTHSEG_APARC_LUT),))
        for sess_id in session_list:
            aff_file = self._get_data(**{'session': sess_id, 'subject': subject, **self.aff_long_ent})
            seg_file = self._get_data(**{'session': sess_id, 'subject': subject, **self.seg_entities})

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
            masks.append(nib.Nifti1Image(mask_arr.astype('uint8'), np.linalg.inv(aff) @ seg_proxy.affine))
            mask_dilated_arr = binary_dilation(mask_arr, ball(3)).astype('float')
            masks_dilated.append(nib.Nifti1Image(mask_dilated_arr, np.linalg.inv(aff) @ seg_proxy.affine))

        # create subject space
        _, template_vox2ras0, template_size = create_empty_template(masks_dilated)
        proxytemplate = nib.Nifti1Image(np.zeros(template_size, dtype='uint8'), template_vox2ras0)
        for mask_proxy in masks:
            template_mask += vol_resample_fast(proxytemplate, mask_proxy) / len(session_list)

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

            session_list = self._get_sessions(subject=subject)
            checkpoint = self._check_running_subject(subject, session_list, force_flag)

            if kwargs.get('verbose', False): print('* Subject: ' + subject)
            if checkpoint['exit_code'] == -1 or checkpoint['exit_code'] == 1:
                if kwargs.get('verbose', False): print(checkpoint['message'])
                return checkpoint

            if checkpoint['exit_code'] in [5]:
                if kwargs.get('verbose', False): print(checkpoint['message'])
                return checkpoint

            if checkpoint['exit_code'] in [0]:
                # initialize graph
                t_cog_d = self._init_graph(subject, session_list, def_dir, force_flag)
                self._update_subject_layout(subject)

                # compute graph
                tmp_dir = join(self.tmp_dir, subject)
                create_dir(tmp_dir)
                graph_kwargs = {'n_epochs': 30, 'cost': 'l1', 'lr': 0.1, 'dir_results': tmp_dir, 'max_iter': 20}
                self._solve_graph(subject, session_list, def_dir, t_cog_d, **graph_kwargs)
                self._update_subject_layout(subject)

            if checkpoint['exit_code'] in [0, 2]:
                # create subject space
                # self._create_subject_space(subject, session_list)
                pr = self._compute_etiv(subject, session_list)
                if pr['exit_code'] != 0:
                    exit_dict = pr
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
            log_keys = ['loss', 'time_duration (s)']
            logger = History(log_keys)
            model_checkpoint = ModelCheckpoint(join(dir_results, 'checkpoints'), -1)
            callbacks = [logger, model_checkpoint]
            if verbose: callbacks += [PrinterCallback()]

            model = InstanceRigidModelLOG(session_list, cost=cost, device=device, reg_weight=0)
            optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr, max_iter=max_iter,
                                          line_search_fn='strong_wolfe')

            min_loss = 1000
            iter_break = 0
            log_dict = {}
            logr = torch.FloatTensor(logr)
            for cb in callbacks:
                cb.on_train_init(model)

            for epoch in range(n_epochs):
                for cb in callbacks:
                    cb.on_epoch_init(model, epoch)

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

                for cb in callbacks:
                    cb.on_step_fi(log_dict, model, epoch, iteration=1, N=1)

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
    def init_st2(timepoints: list[str], input_dir: str, image_shape: tuple[int], factor: int=1,
                 mask_path=None, se=None, penalty=1, dict_flag=False):
        """Load pairwise SVFs and assemble the spanning-tree observation tensors.

        Parameters
        ----------
        timepoints : list of str
            Ordered session IDs.
        input_dir : str
            Directory containing ``<ref>_to_<flo>.nii.gz`` SVF files.
        image_shape : tuple of int
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
        dict_flag : bool, optional
            If ``True``, return dicts keyed by ``'ref_to_flo'`` instead of
            numpy arrays. Default is ``False``.

        Returns
        -------
        phi : np.ndarray or dict
            Pairwise SVF observations.
        obs_mask : np.ndarray or dict
            Per-pair spatial masks.
        w : np.ndarray
            Weight matrix, shape ``(K+1, N)``.
        nk : int
            Number of pairs loaded.
        """
        timepoints_dict = {t: it_t for it_t, t in enumerate(timepoints)}

        N = len(timepoints)
        K = int(N * (N - 1) / 2) + 1
        w = np.zeros((K, N), dtype='int')

        if dict_flag:
            obs_mask = {}
            phi = {}

        else:
            obs_mask = np.zeros(image_shape + (K,))
            phi = np.zeros(image_shape + (3, K,))

        nk = 0
        for tp_ref, tp_flo in itertools.combinations(timepoints, 2):
            t0 = timepoints_dict[tp_ref]
            t1 = timepoints_dict[tp_flo]
            filename = str(tp_ref) + '_to_' + str(tp_flo)

            proxysvf = nib.load(join(input_dir, filename + '.nii.gz'))
            arrsvf = np.asarray(proxysvf.dataobj)

            # Masks
            if mask_path is not None:
                mask_proxy = nib.load(mask_path)
                mask_proxy = vol_resample_fast(proxysvf, mask_proxy)
                mask = np.array(mask_proxy.dataobj)

            else:
                mask = np.ones(image_shape)

            if se is not None:
                mask = binary_dilation(mask, se)

            if dict_flag:
                phi[filename] = factor * arrsvf
                obs_mask[filename] = mask
            else:
                phi[..., nk] = factor * arrsvf
                obs_mask[..., nk] = mask

            w[nk, t0] = -1
            w[nk, t1] = 1
            nk += 1

        if not dict_flag:
            obs_mask[..., nk] = (np.sum(obs_mask[..., :nk - 1]) > 0).astype('uint8')

        w[nk, :] = penalty
        nk += 1
        return phi, obs_mask, w, nk

    @staticmethod
    def st2_L2_global(phi, W, N):
        """Solve the ST² problem globally in L2 via the normal equations.

        Parameters
        ----------
        phi : np.ndarray
            Pairwise SVF observations, shape ``(*image_shape, 3, K)``.
        W : np.ndarray
            Weight matrix, shape ``(K+1, N)``.
        N : int
            Number of timepoints.

        Returns
        -------
        np.ndarray
            Per-timepoint SVFs, shape ``(*image_shape, 3, N)``.
        """
        precision = 1e-6
        lambda_control = np.linalg.inv((W.T @ W) + precision * np.eye(N)) @ W.T
        Tres = lambda_control @ np.transpose(phi, [0, 1, 2, 4, 3])
        Tres = np.transpose(Tres, [0, 1, 2, 4, 3])

        return Tres

    @staticmethod
    def st2_L1(phi, obs_mask, w, N, chunk_id=None, verbose=True):
        """Solve the ST² problem voxel-wise in L1 via linear programming.

        Parameters
        ----------
        phi : np.ndarray
            Pairwise SVF observations, shape ``(*image_shape, 3, K)``.
        obs_mask : np.ndarray
            Per-pair spatial mask, shape ``(*image_shape, K)``.
        w : np.ndarray
            Weight matrix, shape ``(K+1, N)``.
        N : int
            Number of timepoints.
        chunk_id : int, optional
            Chunk identifier printed when processing a spatial sub-block.
        verbose : bool, optional
            If ``True``, print row progress. Default is ``True``.

        Returns
        -------
        np.ndarray
            Per-timepoint SVFs, shape ``(*image_shape, 3, N)``.
        """
        if chunk_id is not None and verbose:
            print("Processing chunk " + str(chunk_id))

        image_shape = obs_mask.shape[:3]
        Tres = np.zeros(image_shape + (3, N))

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
                            c_lp = np.concatenate((np.ones((n_control,)), np.zeros((N,))), axis=0)

                            # Set the inequality
                            A_lp = np.zeros((2 * n_control, n_control + N))
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
    def st2_L1_chunks(phi, obs_mask, w, N, num_chunks=2, num_cores=4):
        """Parallelise :meth:`st2_L1` by dividing the volume into chunks.

        Parameters
        ----------
        phi : np.ndarray
            Pairwise SVF observations, shape ``(*image_shape, 3, K)``.
        obs_mask : np.ndarray
            Spatial mask, shape ``(*image_shape, K)``.
        w : np.ndarray
            Weight matrix, shape ``(K+1, N)``.
        N : int
            Number of timepoints.
        num_chunks : int, optional
            Number of chunks per spatial dimension (total ``num_chunks³``
            jobs). Default is 2.
        num_cores : int, optional
            Parallel workers. ``1`` falls back to serial. Default is 4.

        Returns
        -------
        np.ndarray
            Per-timepoint SVFs, shape ``(*image_shape, 3, N)``.
        """
        if num_cores == 1:
            Tres = USLR_Deformable.st2_L1(phi, obs_mask, w, N)

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
                delayed(USLR_Deformable.st2_L1)(
                    phi[chunk[0], chunk[1], chunk[2]], obs_mask[chunk[0], chunk[1], chunk[2]],
                    w, N, chunk_id=it_chunk) for it_chunk, chunk in enumerate(chunk_list))

            Tres = np.zeros(phi.shape[:4] + (N,))
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

    def _check_running_subject(self, subject: str, session_list: list[str], force_flag: bool=False, register_MNI: bool=False):
        """Determine the processing checkpoint for a subject.

        Parameters
        ----------
        subject : str
            Subject ID.
        timepoints : list of str
            Available session IDs.
        force_flag : bool
            If ``True``, ignore existing outputs.
        register_MNI : bool, optional
            Whether MNI registration is expected. Default is ``False``.

        Returns
        -------
        dict
            ``{'exit_code': int, 'message': str}``. Exit codes:
            ``-1`` error, ``0`` run full pipeline, ``1`` skip,
            ``2`` SVF graph done, ``3`` eTIV done, ``4`` mean SVF done,
            ``5`` single timepoint.
        """
        dict_base = {'subject': subject, 'scope': 'uslr', 'extension': 'nii.gz'}
        dict_svf = {'space': 'uslr', 'task': 'linfit', 'suffix': 'jac', **dict_base}
        dict_MNI = {'space': 'MNI', 'task': 'linfit', 'suffix': 'jac', 'desc': 'mean', **dict_base }

        # do not run if only 1 timepoint available
        if len(timepoints) == 1:
            if register_MNI:
                return {'exit_code': 5, 'message': '[partly done] It has only 1 timepoint. Linking files and registering to MNI \n'}
            else:
                return {'exit_code': 5, 'message': '[done] It has only 1 timepoint. Linking files. \n'}

        # do not run if only 0 timepoint available
        elif len(timepoints) == 0:
            return {'exit_code': 1, 'message': '[done] It has 0 timepoints available. Skipping.\n'}

        # check if all timepoints are linearly registered
        elif any([self._get_data(**{'subject': subject, 'session': t, **self.im_long_entities}) is None
                  for t in timepoints]):
            return {'exit_code': -1, 'message': '[error] not all timepoints are correctly registered in uslr. '
                                                'Please check. This may be to preprocessing errors.\n'}

        # check if the graph has been solved
        elif (self._get_data(**{'subject': subject, 'session': timepoints, **self.svf_long_entities}, curr_len=len(timepoints), verbose=False) is not None and not force_flag):

            if not exists(join(DIR_PIPELINES[self.pipeline_dir], 'sub-' + subject, 'sub-' + subject + '_T1wetiv.npy')):
                return {'exit_code': 2,
                        'message': '[partly done] graph and template are already computed.\n'}

            elif (self._get_data(**dict_svf, verbose=False) is None):
                return {'exit_code': 3,
                        'message': '[partly done] graph, template and etiv done. Computing mean trajectories.\n'}

            elif (self._get_data(**dict_MNI, verbose=False) is None and register_MNI is True):
                return {'exit_code': 4,
                        'message': '[partly done] graph, template, etiv and mean SVF done; MNI registration is missing.\n'}

            else:
                return {'exit_code': 1,
                        'message': '[done] subject already processed. '
                                   'Check the results in [..]/uslr/sub-' + subject + '.\n'}

        # do not run if more segmentations than timepoints are found
        elif self._get_data(**{'subject': subject, **self.seg_entities}, curr_len=len(timepoints)) is None:
            return {'exit_code': -1,
                    'message': '[error] not all timepoints are segmented. Please, run preprocess/synthseg.py first.\n'}

        else:
            return {'exit_code': 0, 'message': 'running USLR'}

    def _init_graph(self,subject: str, session_list: list[str], def_dir, force_flag=False):
        """Register all pairs of timepoints with SynthMorph and save the SVFs.

        Parameters
        ----------
        subject : str
            Subject ID.
        timepoints : list of str
            Session IDs.
        def_dir : str
            Directory where ``<ref>_to_<flo>.nii.gz`` SVF files are written.
        force_flag : bool, optional
            If ``True``, recompute even when files exist. Default is
            ``False``.
        """
        svf_v2r = np.load(self._get_data(**{'subject': subject, **self.svf_v2r_entities}).path)
        for tp_ref, tp_flo in itertools.permutations(timepoints, 2):
            output_filepath = join(def_dir, str(tp_ref) + '_to_' + str(tp_flo) + '.nii.gz')
            if exists(output_filepath) and not force_flag:
                continue

            # read image and mask
            imageref_file = self._get_data(**{'subject': subject, 'session': tp_ref, **self.im_long_entities})
            imageflo_file = self._get_data(**{'subject': subject, 'session': tp_flo, **self.im_long_entities})

            maskref_file = self._get_data(**{'subject': subject, 'session': tp_ref, **self.mask_long_entities})
            maskflo_file = self._get_data(**{'subject': subject, 'session': tp_flo, **self.mask_long_entities})

            if imageref_file is None or imageflo_file is None or maskref_file is None or maskflo_file is None:
                continue

            fw_svf = synthmorph_register(imageref_file, imageflo_file)

            img = nib.Nifti1Image(fw_svf, svf_v2r)
            nib.save(img, output_filepath)

    def _solve_graph(self, subject: str, session_list: list[str], def_dir, cost, **kwargs):
        """Solve the deformable spanning-tree problem with BCH approximation.

        Parameters
        ----------
        subject : str
            Subject ID.
        timepoints : list of str
            Session IDs.
        def_dir : str
            Directory with pairwise SVF NIfTI files.
        cost : {'bch-l1', 'bch-l2', 'l1', 'l2'}
            Optimisation strategy: BCH variants use the additive SVF
            approximation; plain L1/L2 use linear programming.

        Returns
        -------
        dict
            ``{session_id: np.ndarray}`` per-timepoint SVFs,
            shape ``(*svf_shape, 3)``.
        """
        R, M, W, NK = USLR_Deformable.init_st2(timepoints, def_dir, self.svf_shape, se=None, dict_flag=False)

        if cost == 'bch-l2':
            T_latent = USLR_Deformable.st2_L2_global(R, W, len(timepoints))
            T_latent = {t: T_latent[..., it_t] for it_t, t in enumerate(timepoints)}

        else:
            T_latent = USLR_Deformable.st2_L1_chunks(R, M, W, len(timepoints), num_cores=1)
            T_latent = {t: T_latent[..., it_t] for it_t, t in enumerate(timepoints)}

        return T_latent

    def _compute_template(self, subject: str, session_list: list[str], **kwargs):
        """Build the nonlinear template image, segmentation, mask, and eTIV.

        Warps each timepoint to the linear template space using the estimated
        SVFs, takes the median image and majority-vote segmentation, and
        saves the results to the ``uslr`` derivative.

        Parameters
        ----------
        subject : str
            Subject ID.
        timepoints : list of str
            Session IDs.
        **kwargs
            Ignored.
        """
        sss_file = self._get_data(**{'subject': subject, 'session': None, 'scope':'uslr-lin', **self.template_long_entities})
        if sss_file is None:
            return

        proxyref = nib.load(sss_file.path)

        # build path template: image, mask, seg
        image_filename = self.build_path({'suffix': 'T1w', 'subject': subject, **self.template_nonlin_entities})
        image_std_filename = self.build_path({'suffix': 'T1wstd', 'subject': subject, **self.template_nonlin_entities})
        mask_filename = self.build_path({'suffix': 'T1wmask', 'subject': subject, **self.template_nonlin_entities})
        seg_filename = self.build_path({'suffix': 'T1wdseg', 'subject': subject, **self.template_nonlin_entities})

        # compute template: image, mask, seg
        image_list = []
        seg_list = []
        for tp in timepoints:
            im_file = self._get_data(**{'subject': subject, 'session': tp, **self.bf_entities})
            seg_file = self._get_data(**{'subject': subject, 'session': tp, **self.seg_entities})
            aff_file = self._get_data(**{'subject': subject, 'session': tp, **self.aff_graph_entities})
            svf_file = self._get_data(**{'subject': subject, 'session': tp, **self.svf_graph_entities})

            if svf_file is None or aff_file is None or im_file is None or seg_file is None:
                continue

            proxyimage = nib.load(im_file.path)
            proxyseg = nib.load(seg_file.path)
            aff = np.load(aff_file.path)
            proxysvf = nib.load(svf_file.path)
            flow = integrate_svf(np.array(proxysvf.dataobj), self.net_shape, scaling_factor=2, int_steps=7)
            proxyflow = nib.Nifti1Image(flow, affine=proxyref.affine)

            # Image
            arrayim = np.array(proxyimage.dataobj)
            arrayim = gaussian_antialiasing(arrayim, proxyimage.affine, [1, 1, 1])
            proxyimage = nib.Nifti1Image(arrayim, np.linalg.inv(aff) @ proxyimage.affine)
            proxyimage = vol_resample_fast(proxyref, proxyimage, proxyflow=proxyflow)
            proxyimage.uncache()

            arrayseg = np.array(proxyseg.dataobj)
            arrayonehot = one_hot_encoding(arrayseg, categories=SYNTHSEG_APARC_LUT).astype('float')
            proxyonehot = nib.Nifti1Image(arrayonehot, np.linalg.inv(aff) @ proxyseg.affine)
            proxyonehot = vol_resample_fast(proxyref, proxyonehot, proxyflow=proxyflow)
            proxyonehot.uncache()

            image_list.append(proxyimage)
            seg_list.append(proxyonehot)

        # save image (median and std), mask (and etiv) and seg.
        arr_image_list = np.stack([np.array(proxyim.dataobj) for proxyim in image_list], axis=0)
        template = np.median(arr_image_list, axis=0)
        template_std = np.std(arr_image_list, axis=0)
        del arr_image_list

        template_seg = np.zeros(template.shape + (len(SYNTHSEG_APARC_LUT),))
        for proxyseg in seg_list:
            template_seg += np.array(proxyseg.dataobj)

        template_seg = np.argmax(template_seg, axis=-1)
        template_seg = self._undo_one_hot(template_seg)
        template_mask = (template_seg > 0) & (template_seg != 24)
        etiv = np.sum(template_seg > 0)

        save_volume(template, proxyref.affine, join(DIR_PIPELINES['uslr'], image_filename))
        save_volume(template_std, proxyref.affine, join(DIR_PIPELINES['uslr'], image_std_filename))
        save_volume(template_seg, proxyref.affine, join(DIR_PIPELINES['uslr'], seg_filename))
        save_volume(template_mask, proxyref.affine, join(DIR_PIPELINES['uslr'], mask_filename))

        sid = 'sub-' + str(subject)
        np.save(join(DIR_PIPELINES['uslr'], sid, sid + '_T1wetiv.npy'), etiv)

    def _compute_mean_svf(self, subject: str, session_list: list[str], **kwargs):
        """Fit a linear trajectory through per-timepoint SVFs and save statistics.

        Runs ordinary least squares over time to decompose per-timepoint
        SVFs into an intercept and a slope (rate-of-change SVF). Also
        integrates the slope SVF and saves the Jacobian determinant map.

        Parameters
        ----------
        subject : str
            Subject ID.
        timepoints : list of str
            Session IDs.
        **kwargs
            Ignored.
        """
        fit_entities = {'subject': subject, 'space': 'uslr', 'task': 'linfit', 'extension': 'nii.gz', 'scope': 'uslr', 'suffix': 'svf'}
        image_filename = self.build_path({'desc': 'mean', **fit_entities})
        error_filename = self.build_path({'desc': 'error', **fit_entities})
        std_filename = self.build_path({'desc': 'std', **fit_entities})

        fit_entities['suffix'] = 'def'
        error_flow_filename = self.build_path({'desc': 'error', **fit_entities})
        std_flow_filename = self.build_path({'desc': 'std', **fit_entities})

        fit_entities['suffix'] = 'jac'
        jac_filename = self.build_path({'desc': 'mean', **fit_entities})

        linreg = LinearRegression()
        time_list = self._get_time_list(subject, timepoints)

        file_v2r = self._get_data(**{'subject': subject, **self.svf_v2r_entities})
        if file_v2r is None:
            return

        svf_v2r = np.load(file_v2r.path)

        svf_list = []
        features_list = []
        for tp in timepoints:
            svf_file = self._get_data(**{'subject': subject, 'session': tp, **self.svf_graph_entities})
            if svf_file is None:
                continue

            proxysvf = nib.load(svf_file.path)
            svf_list.append(np.array(proxysvf.dataobj).reshape(-1))

            age = float(time_list[tp])
            features_list.append([age])

        X = np.array(features_list)
        Y = np.stack(svf_list, axis=0)
        linreg.fit(X, Y)

        coef_list = [linreg.coef_[:, it_f].reshape(self.svf_shape + (3,)) for it_f in range(len(features_list[0]))]
        intercept_list = [linreg.intercept_.reshape(self.svf_shape + (3,))]
        results_vol = np.stack(intercept_list + coef_list, axis=-1)

        y_pred = linreg.predict(X)
        error_vol = np.sum((Y - y_pred) ** 2 / len(svf_list), axis=0).reshape(self.svf_shape + (3,))
        std_vol = np.sum((Y - np.mean(Y, axis=0)) ** 2 / len(svf_list), axis=0).reshape(self.svf_shape + (3,))

        save_volume(results_vol, svf_v2r, join(DIR_PIPELINES['uslr'], image_filename))
        save_volume(error_vol, svf_v2r, join(DIR_PIPELINES['uslr'], error_filename))
        save_volume(std_vol, svf_v2r, join(join(DIR_PIPELINES['uslr'], std_filename)))

        # Flow error
        flow_pred = []
        flow = []
        for it_tp in range(Y.shape[0]):
            svf_pred = y_pred[it_tp].reshape(self.svf_shape + (3,))
            svf = Y[it_tp].reshape(self.svf_shape + (3,))
            flow_pred += [integrate_svf(svf_pred, self.net_shape, scaling_factor=2, int_steps=7)]
            flow += [integrate_svf(svf, self.net_shape, scaling_factor=2, int_steps=7)]

        flow_pred = np.stack(flow_pred, axis=0)
        flow = np.stack(flow, axis=0)
        flow_error_vol = np.abs(flow - flow_pred).reshape((len(flow),) + self.net_shape + (3,))
        flow_std_vol = (flow - np.mean(flow, axis=0) ** 2).reshape((len(flow),) + self.net_shape + (3,))

        save_volume(flow_error_vol, svf_v2r, join(DIR_PIPELINES['uslr'], error_flow_filename))
        save_volume(flow_std_vol, svf_v2r, join(join(DIR_PIPELINES['uslr'], std_flow_filename)))

        net_v2r = np.load(self._get_data(**{'subject': subject, **self.net_v2r_entities}).path)
        svf = results_vol[..., 1]
        if max(time_list.values()) - min(time_list.values()) > 30:
            svf = svf*365.25

        flow = integrate_svf(svf, self.net_shape, scaling_factor=2, int_steps=7)
        jac = compute_jacobian(flow)
        save_volume(jac, net_v2r, join(DIR_PIPELINES['uslr'], jac_filename))


        seg_filename = self.build_path({'suffix': 'T1wdseg', 'subject': subject, **self.template_nonlin_entities})
        proxyseg = nib.load(join(DIR_PIPELINES['uslr'], seg_filename))
        seg_cort = np.array(proxyseg.dataobj)

        proxyseg = nib.load(join(DIR_PIPELINES['uslr-lin'], seg_filename))
        seg_subcort = np.array(proxyseg.dataobj)

        filepath = join(DIR_PIPELINES['uslr'], 'sub-' + str(subject), 'sub-' + str(subject) + '_jac.csv')

        if exists(filepath):
            data_df = pd.read_csv(filepath, dtype=str)
        else:
            data_df = pd.DataFrame(columns=['metric'] + list(self.labels_dict.values()))

        for mn, mf in {'mean': np.mean, 'median': np.median, 'std': np.std, 'min': np.min, 'max': np.max}.items():
            d = {'metric': [mn]}
            for lnum, lname in self.labels_dict.items():
                if 'ctx' in lname:
                    seg = seg_cort
                else:
                    seg = seg_subcort

                jval = jac[seg == lnum]
                if lnum in seg:
                    d[lname] = [mf(jval)]
                else:
                    d[lname] = [np.nan]

            data_df = pd.concat([data_df, pd.DataFrame(d)], ignore_index=True)

        data_df.to_csv(filepath, index=False)

    # def _register_to_MNI(self, subject, **kwargs):
    #     """Register the nonlinear template to MNI via centroid affine + SynthMorph.
    #
    #     Saves the affine, SVF, and Jacobian determinant in MNI space.
    #
    #     Parameters
    #     ----------
    #     subject : str
    #         Subject ID.
    #     **kwargs
    #         Ignored.
    #     """
    #     scope = 'uslr'
    #
    #     jac_entities = {'subject': subject, 'task': 'linfit', 'scope': scope, 'suffix': 'jac', 'desc': 'mean'}
    #     mni_entities = {'subject': subject, 'space': 'MNI', 'desc': 'tosubject'}
    #     aff_fname = self.build_path({'suffix': 'aff', 'extension': 'npy', **mni_entities})
    #     svf_fname = self.build_path({'suffix': 'svf', 'extension': 'nii.gz', **mni_entities})
    #     v2r_fname = self.build_path({'suffix': 'v2r', 'extension': 'npy', **mni_entities})
    #
    #     template_im = self._get_data(**{'suffix': 'T1w', 'subject': subject, **self.template_nonlin_entities})
    #     template_seg = self._get_data(**{'suffix': 'T1wdseg', 'subject': subject, **self.template_nonlin_entities})
    #     if template_seg is None or template_im is None:
    #         return
    #
    #     centroid_ref, ok_ref = compute_centroids_ras(MNI_TEMPLATE_SEG, labels_registration)
    #     centroid_flo, ok_flo = compute_centroids_ras(template_seg.path, labels_registration)
    #
    #     M_sbj = getM(centroid_ref[:, ok_ref > 0], centroid_flo[:, ok_ref > 0], use_L1=False)
    #     np.save(join(DIR_PIPELINES[scope], aff_fname), M_sbj)
    #
    #     sfmni = sf.load_volume(MNI_TEMPLATE)
    #     net2vox, vox2net, net_v2r = network_space(sfmni, shape=self.net_shape, center=sfmni)
    #     np.save(join(DIR_PIPELINES[scope], v2r_fname), net_v2r)
    #     svf_v2r = net_v2r.copy()
    #     for c in range(3):
    #         svf_v2r[:-1, c] = svf_v2r[:-1, c] / 0.5
    #     svf_v2r[:-1, -1] = svf_v2r[:-1, -1] - np.matmul(svf_v2r[:-1, :-1], 0.5 * (np.array([0.5] * 3) - 1))
    #
    #     proxynet = nib.Nifti1Image(np.zeros(self.net_shape, dtype='float32'), net_v2r)
    #     proxytemplate = nib.load(MNI_TEMPLATE)
    #     proxytemplate = vol_resample_fast(proxynet, proxytemplate)
    #
    #     proxysubject = nib.load(template_im)
    #     arrsubject = np.array(proxysubject.dataobj)
    #     proxysubject = nib.Nifti1Image(arrsubject, np.linalg.inv(M_sbj) @ proxysubject.affine)
    #     proxysubject = vol_resample_fast(proxynet, proxysubject)
    #
    #     fw_svf = synthmorph_register(proxytemplate, proxysubject, reg_param=0.4)
    #     save_nii(fw_svf, svf_v2r, join(DIR_PIPELINES[scope], svf_fname))
    #
    #     jac_entities['space'] = 'uslr'
    #     jac_file = self._get_data(**jac_entities)
    #     if jac_file is not None:
    #         proxyjac = nib.load(jac_file.path)
    #         arrjac = np.array(proxyjac.dataobj)
    #         proxyjac = nib.Nifti1Image(arrjac, np.linalg.inv(M_sbj) @ proxyjac.affine)
    #         flow = integrate_svf(fw_svf, self.net_shape, scaling_factor=2, int_steps=7)
    #         proxyflow = nib.Nifti1Image(flow, net_v2r)
    #         proxysubject = vol_resample_fast(proxynet, proxyjac, proxyflow=proxyflow)
    #         jac_entities['space'] = 'MNI'
    #         jac_filename = self.build_path(jac_entities)
    #         nib.save(proxysubject, join(DIR_PIPELINES[scope], jac_filename))

    def process_subject(self, subject, cost=Literal['bch-l1', 'bch-l2'], force_flag: bool=False, **kwargs):
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
        print('* Subject: ' + subject)
        assert cost in ['bch-l1', 'bch-l2']
        self.svf_long_entities['scope'] = 'nicgiprep-long'

        session_list = self._get_sessions(subject=subject, uslr=False)
        checkpoint = self._check_running_subject(subject, session_list, force_flag)
        print('  -->', checkpoint['message'])
        if checkpoint['exit_code'] == -1 or checkpoint['exit_code'] == 1:
            return

        def_dir = join(self.tmp_dir, 'sub-' + subject)
        create_dir(def_dir)

        if checkpoint['exit_code'] in [0]:
            # compute svf v2r
            svf_v2r_file = self._get_data(subject=subject, **self.svf_v2r_entities)
            if svf_v2r_file is None:
                net_v2r = np.load(self._get_data(subject=subject, **self.net_v2r_entities).path)
                svf_v2r = net_v2r.copy()
                for c in range(3):
                    svf_v2r[:-1, c] = svf_v2r[:-1, c] / 0.5
                svf_v2r[:-1, -1] = svf_v2r[:-1, -1] - np.matmul(svf_v2r[:-1, :-1], 0.5 * (np.array([0.5] * 3) - 1))

                filename_v2r = self.build_path({'subject': subject, **self.svf_v2r_entities})
                np.save(join(DIR_PIPELINES['uslr-lin'], str(filename_v2r)), svf_v2r)

            else:
                svf_v2r = np.load(svf_v2r_file.path)

            # build the entire graph
            self._init_graph(subject, session_list, def_dir, force_flag)
            self._update_subject_layout(subject)

            # solve spanning tree
            T_latent = self._solve_graph(subject, session_list, def_dir, cost, **kwargs)
            for sess_id in session_list:
                filename = self.build_path({'subject': subject, 'session': sess_id, **self.svf_long_entities})
                create_dir(dirname(join(DIR_PIPELINES['nicgiprep-long'], filename)))

                img = nib.Nifti1Image(T_latent[sess_id].astype('float32'), svf_v2r)
                save_volume(T_latent[sess_id].astype('float32'), svf_v2r, path=join(DIR_PIPELINES['uslr'], filename))

            self._update_subject_layout(subject)

        if checkpoint['exit_code'] in [0, 2]:
            # create subject space
            self._compute_template(subject, session_list)
            self._update_subject_layout(subject)

        if checkpoint['exit_code'] in [0, 2, 3]:
            # compute mean SVF
            self._compute_mean_svf(subject, session_list)
            self._update_subject_layout(subject)

        # if checkpoint['exit_code'] in [0, 2, 3, 4, 5]:
        #     # register to MNI
        #     self._register_to_MNI(subject)

        print('DONE. \n')


class LongitudinalRegistration(LongitudinalProcessor):
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


    def process_subject(self, subject: str, force_flag: bool=False, **kwargs) -> ProcessResult:

        # call linear --> save etiv and T1w in subject space
        # call nonlinear --> save nonlinear templates (T1w, T1wdseg) and SVF and JAC.
        # deform nonlinear template to MNI
        # propagate all to MNI.
        return


class LongitudinalSegmentationProcessor(LongitudinalProcessor):
    """Processing subclass for longitudinal segmentation and volumetry.

    Adds helpers for computing and saving volumetric measurements from
    hard segmentations and soft posteriors.
    """

    def __init__(self, bids_loader, subject_list=None, pipeline_dir=None, **kwargs):
        """
        Parameters
        ----------
        bids_loader : BIDSLayout
            Initialised PyBIDS layout.
        subject_list : list of str, optional
            Subject IDs to process. Defaults to all subjects.
        pipeline_dir : str, optional
            Output pipeline directory key (looked up in ``DIR_PIPELINES``).
        **kwargs
            Forwarded to :class:`Processing`.
        """
        self._pipeline_dir = pipeline_dir
        super(LongitudinalSegmentationProcessor, self).__init__(bids_loader=bids_loader, subject_list=subject_list,
                                                                 **kwargs)

    def _name(self):
        """Return the display name of this pipeline."""
        return 'LongitudinalSegmentation'

    @property
    def pipeline_dir(self):
        """Output pipeline directory key.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this property.
        """
        raise NotImplementedError("The pipeline_dir property should be implemented by sub-classes")

    def _save_vols(self, vols, filepath, labels_lut=None, *args, **kwargs):
        """Append volumetric measurements to a per-subject CSV file.

        Existing rows for the same ``(session, method)`` key are dropped and
        replaced.

        Parameters
        ----------
        vols : dict
            Nested dict ``{session: {method: {channel_index: volume_mm3}}}``.
        filepath : str
            Path to the output CSV file (created if it does not exist).
        labels_lut : dict, optional
            Label LUT mapping integer label IDs to channel indices. Defaults
            to ``self.labels_lut``.
        """
        # filepath = join(DIR_PIPELINES[self.pipeline_dir], 'sub-' + subject, 'sub-' + subject + '_vols.csv')
        if labels_lut is None:
            labels_lut = self.labels_lut

        if exists(filepath):
            data_df = pd.read_csv(filepath)
        else:
            data_df = pd.DataFrame(columns=['session', 'method'] + list(self.labels_dict.values()))

        data_df.set_index(['session', 'method'], drop=False, inplace=True)
        for tp, tp_dict in vols.items():
            for method, method_dict in tp_dict.items():
                if (tp, method) in data_df.index:
                    data_df.drop((tp, method), inplace=True)

                v_dict = {v: method_dict[labels_lut[k]] if labels_lut[k] in method_dict.keys() else 0 for k, v in
                          self.labels_dict.items() if k in labels_lut.keys()}
                df = pd.Series({'session': tp, 'method': method, **v_dict})
                data_df = pd.concat([data_df, df.to_frame().T], ignore_index=True)
                data_df.set_index(['session', 'method'], drop=False, inplace=True)

        data_df.set_index('session', inplace=True, drop=False)
        data_df.to_csv(filepath, index=False)

    def _get_vols(self, y, res=1, labels=None):
        """Compute label volumes from a hard segmentation.

        Parameters
        ----------
        y : np.ndarray
            Integer label map.
        res : float or list of float, optional
            Voxel size in mm (isotropic or per-axis). Default is 1.
        labels : array-like, optional
            Label values to compute volumes for. Defaults to
            ``np.unique(y)``.

        Returns
        -------
        dict
            Mapping ``{label_int: volume_mm3}``.
        """
        if labels is None:
            labels = np.unique(y)

        n_dims = len(y.shape)
        if isinstance(res, int):
            res = [res] * n_dims
        vol_vox = np.prod(res)

        vols = {}
        for l in labels:
            mask_l = y == l
            vols[int(l)] = np.round(np.sum(mask_l) * vol_vox, 2)

        return vols

    def _get_vols_post(self, post, res=1):
        """Compute expected label volumes from soft posteriors.

        Normalises posteriors to sum to one, then sums each channel over
        the spatial dimensions.

        Parameters
        ----------
        post : np.ndarray
            Soft segmentation, shape ``(*spatial, n_labels)``.
        res : float or list of float, optional
            Voxel size in mm. Default is 1.

        Returns
        -------
        dict
            Mapping ``{channel_index: expected_volume_mm3}``.
        """
        n_labels = post.shape[-1]
        n_dims = len(post.shape[:-1])
        if isinstance(res, int):
            res = [res] * n_dims
        vol_vox = np.prod(res)

        post /= np.sum(post, axis=-1, keepdims=True)

        vols = {}
        for l in range(n_labels):
            mask_l = post[..., l]
            vols[l] = np.round(np.sum(mask_l) * vol_vox, 2)

        return vols

    def _undo_one_hot(self, y, labels_lut=None, dtype='float32'):
        """Convert a one-hot channel index array back to integer label values.

        Parameters
        ----------
        y : np.ndarray
            Array of channel indices.
        labels_lut : dict, optional
            Label LUT. Defaults to ``self.labels_lut``.
        dtype : str, optional
            Output NumPy dtype. Default is ``'float32'``.

        Returns
        -------
        np.ndarray
            Integer label map, same shape as ``y``.
        """
        if labels_lut is None:
            labels_lut = self.labels_lut

        y_true = np.zeros_like(y)
        for ul, it_ul in labels_lut.items():
            y_true[y == it_ul] = ul

        return y_true.astype(dtype)

class LongLabelFusionProcessor(LongitudinalSegmentationProcessor):
    """Label-fusion pipeline for longitudinal segmentation.

    Warps each timepoint's image and one-hot segmentation to every other
    timepoint's space and fuses them with optional temporal and appearance
    weighting kernels.
    """

    def _name(self):
        """Return the display name of this pipeline."""
        return 'LabelFusion'

    def _get_tp_displacement(self, subject, target_tp, atlas_tp, **kwargs):
        """Return the dense displacement field mapping ``atlas_tp`` to ``target_tp``.

        Parameters
        ----------
        subject : str
            Subject ID.
        target_tp : str
            Session ID of the target (reference) timepoint.
        atlas_tp : str
            Session ID of the atlas (moving) timepoint.
        **kwargs
            Pipeline-specific arguments.

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
        """
        raise NotImplementedError

    def _deform_atlases(self, subject, target_tp, timepoints, results_dir, *args, **kwargs):
        """Warp all atlas timepoints to the target timepoint space and save them.

        For each atlas timepoint, resamples the T1w image and the one-hot
        segmentation to the target space using the displacement returned by
        :meth:`_get_tp_displacement`. The target timepoint itself is saved
        as-is (no warping).

        Parameters
        ----------
        subject : str
            Subject ID.
        target_tp : str
            Session ID of the target timepoint.
        timepoints : list of str
            All session IDs (including the target).
        results_dir : str
            Directory where warped images (``<tp>.im.nii.gz``) and one-hot
            arrays (``<tp>.onehot.nii.gz``) are written.
        """
        seg_ref_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': target_tp})
        if seg_ref_file is None:
            return

        proxysegref = nib.load(seg_ref_file.path)
        for atlas_tp in timepoints:
            im_file = self._get_data(**{**self.bf_entities, 'subject': subject, 'session': atlas_tp})
            seg_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': atlas_tp})

            if im_file is None or seg_file is None:
                continue

            output_image_filepath = join(results_dir, str(atlas_tp) + '.im.nii.gz')
            output_onehot_filepath = join(results_dir, str(atlas_tp) + '.onehot.nii.gz')

            # One-hot encoding of the labels
            proxyseg = nib.load(seg_file.path)
            seg_arr = np.array(proxyseg.dataobj)
            onehot_arr = one_hot_encoding(seg_arr, categories=self.labels_lut)

            # Gaussian filter for [1, 1, 1] segmentation
            proxyim = nib.load(im_file.path)
            arrayim = np.array(proxyim.dataobj)
            arrayim = gaussian_antialiasing(arrayim, proxyim.affine, [1, 1, 1])
            proxyim = nib.Nifti1Image(arrayim, proxyim.affine)
            proxyim = vol_resample_fast(proxyseg, proxyim)
            arrayim = np.array(proxyim.dataobj)

            if target_tp == atlas_tp:
                proxyonehot = nib.Nifti1Image(onehot_arr.astype('float32'), proxyseg.affine)
                nib.save(proxyonehot, output_onehot_filepath)
                nib.save(proxyim, output_image_filepath)

            else:
                tp_displ = self._get_tp_displacement(subject, target_tp, atlas_tp, **kwargs)

                # Image
                arrayim_mov = warp(arrayim, tp_displ)
                proxyim = nib.Nifti1Image(arrayim_mov.astype('float32'), proxysegref.affine)
                nib.save(proxyim, output_image_filepath)

                arrayonehot_mov = warp(onehot_arr, tp_displ)
                proxyonehot = nib.Nifti1Image(arrayonehot_mov.astype('float32'), proxysegref.affine)
                nib.save(proxyonehot, output_onehot_filepath)

    def _label_fusion(self, target_tp, timepoints, results_dir, time_scale=None, g_std=None, **kwargs):
        """Fuse warped atlas one-hot arrays at the target timepoint.

        Computes a weighted average of all available atlas one-hot arrays,
        with optional appearance (Gaussian) and temporal (exponential decay)
        kernels.

        Parameters
        ----------
        target_tp : str
            Session ID of the target timepoint.
        timepoints : list of str
            All session IDs to include as atlases.
        results_dir : str
            Directory containing pre-warped atlas files (``<tp>.im.nii.gz``
            and ``<tp>.onehot.nii.gz``).
        time_scale : float, optional
            Temporal kernel scale (exponential decay). ``None`` disables
            temporal weighting.
        g_std : float, optional
            Appearance kernel standard deviation (Gaussian on intensity
            difference). ``None`` disables appearance weighting.
        **kwargs
            Must contain ``'time_list'`` (dict) when ``time_scale`` is set.

        Returns
        -------
        seg : np.ndarray
            Hard segmentation (argmax), same spatial shape as target.
        posteriors : np.ndarray
            Soft fused posteriors, shape ``(*spatial, n_labels)``.
        affine : np.ndarray
            Affine matrix of the target image, shape ``(4, 4)``.
        """
        image_targ_filepath = join(results_dir, str(target_tp) + '.im.nii.gz')
        proxyim_targ = nib.load(image_targ_filepath)
        arrayim_targ = np.array(proxyim_targ.dataobj)
        # arrayim_targ = gaussian_antialiasing(arrayim_targ, proxyim_targ.affine, [1, 1, 1])

        arrayonehot_targ = None
        for atlas_tp in timepoints:
            image_filepath = join(results_dir, str(atlas_tp) + '.im.nii.gz')
            onehot_filepath = join(results_dir, str(atlas_tp) + '.onehot.nii.gz')
            if not exists(image_filepath) or not exists(onehot_filepath):
                continue

            proxyim = nib.load(image_filepath)
            arrayim = np.array(proxyim.dataobj)

            if g_std == None:
                g_ker = 1
            else:
                g_ker = 1 / np.sqrt(2 * np.pi) / g_std * np.exp(-0.5 / (g_std ** 2) * (arrayim_targ - arrayim) ** 2)

            if time_scale == None:
                t_ker = 1
            else:
                t_targ = kwargs['time_list'][target_tp]
                t_atlas = kwargs['time_list'][atlas_tp]
                t_ker = time_scale * np.exp(-time_scale * (t_targ - t_atlas))

            pdata = t_ker * g_ker
            if g_std != None or time_scale != None:
                pdata = pdata[..., np.newaxis]

            proxyonehot = nib.load(onehot_filepath)
            arrayonehot = np.array(proxyonehot.dataobj)

            if arrayonehot_targ is None:
                arrayonehot_targ = np.zeros(arrayonehot.shape)

            arrayonehot_targ += pdata * arrayonehot

        mask = np.sum(arrayonehot_targ[..., 1:], axis=-1) > 0
        arrayonehot_targ[~mask, 0] = 1
        arrayonehot_targ /= np.sum(arrayonehot_targ, axis=-1, keepdims=True)
        return np.argmax(arrayonehot_targ, axis=-1), arrayonehot_targ, proxyim_targ.affine

    def _save_vols(self, vols, filepath, time_scale=None, g_std=None, labels_lut=None, **kwargs):
        """Append label-fusion volumetric results to a CSV file.

        Extends :meth:`LongitudinalSegmentationProcessing._save_vols` with
        extra columns for ``time_scale`` and ``g_std`` kernel parameters.

        Parameters
        ----------
        vols : dict
            Nested dict ``{session: {method: {channel_index: volume_mm3}}}``.
        filepath : str
            Path to the output CSV.
        time_scale : float or None, optional
            Temporal kernel parameter recorded in the CSV.
        g_std : float or None, optional
            Appearance kernel parameter recorded in the CSV.
        labels_lut : dict, optional
            Label LUT. Defaults to ``self.labels_lut``.
        """
        if labels_lut is None:
            labels_lut = self.labels_lut

        if exists(filepath):
            data_df = pd.read_csv(filepath, dtype=str)
        else:
            data_df = pd.DataFrame(
                columns=['session', 'method', 'time_scale', 'g_std'] + list(self.labels_dict.values()))

        data_df.set_index(['session', 'method'], drop=False, inplace=True)
        for tp, tp_dict in vols.items():
            for method, method_dict in tp_dict.items():
                if (tp, method) in data_df.index:
                    data_df.drop((tp, method), inplace=True)

                vols = {v: method_dict[labels_lut[k]] if labels_lut[k] in method_dict.keys() else 0 for k, v in
                        self.labels_dict.items()}
                df = pd.Series({'session': tp, 'method': method, 'time_scale': time_scale, 'g_std': g_std, **vols})
                data_df = pd.concat([data_df, df.to_frame().T], ignore_index=True)
                data_df.set_index(['session', 'method'], drop=False, inplace=True)

        data_df.set_index('session', inplace=True, drop=False)
        data_df.to_csv(filepath, index=False)

    def process_subject(self, subject, force_flag=False, *args, **kwargs):
        """Run label fusion and volumetry for all timepoints of one subject.

        Iterates over each target timepoint, deforms all other timepoints'
        atlases to the target space, fuses them, saves the hard segmentation
        and a volumetric CSV.

        Parameters
        ----------
        subject : str
            Subject ID.
        force_flag : bool, optional
            If ``True``, reprocess even when volumes CSV already exists.
            Default is ``False``.
        *args
            Unused positional arguments.
        **kwargs
            Forwarded to :meth:`_deform_atlases` and :meth:`_label_fusion`.
            Should include ``'time_list'`` (dict) when temporal kernels are
            used.
        """
        print('\nSubject: ' + subject)
        timepoints = self._get_timepoints(subject=subject, uslr=True)

        kwargs['time_list'] = self._get_time_list(subject, timepoints)

        if 'output_pipeline' in kwargs.keys():
            output_pipeline = kwargs['output_pipeline']
        else:
            output_pipeline = self.pipeline_dir

        vols_filepath = join(DIR_PIPELINES[output_pipeline], 'sub-' + subject, 'sub-' + subject + '_vols.csv')
        if exists(vols_filepath) and not force_flag:
            return

        # Save image and label registration
        sbj_vols = {}
        print(' * Timepoint: ', end=' ', flush=True)
        for target_tp in timepoints:
            print(target_tp, end='', flush=True)

            # I/O data
            seg_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': target_tp})
            if seg_file is None:
                continue

            tmp_dir = join(self.tmp_dir, subject, str(target_tp))
            create_dir(tmp_dir)

            filename = self.bids_loader.build_path({'subject': subject, 'session': target_tp, 'extension': 'nii.gz',
                                                    'suffix': 'T1wdseg', 'acquisition': '1'},
                                                   absolute_paths=False, path_patterns=BIDS_PATH_PATTERN,
                                                   strict=False, validate=False)
            create_dir(dirname(join(DIR_PIPELINES[output_pipeline], filename)))

            # Main processes
            self._deform_atlases(subject, target_tp, timepoints, tmp_dir, **kwargs)
            seg, posteriors, v2r = self._label_fusion(target_tp, timepoints, tmp_dir, **kwargs)

            # Store results
            seg_true = self._undo_one_hot(seg, dtype='int16')
            img = nib.Nifti1Image(seg_true, v2r)
            nib.save(img, join(DIR_PIPELINES[output_pipeline], filename))

            pixdim = np.sqrt(np.sum(v2r * v2r, axis=0))[:-1]
            vols = self._get_vols(seg, res=pixdim, labels=list(self.labels_lut.values()))
            vols_post = self._get_vols_post(posteriors, res=pixdim)
            sbj_vols[target_tp] = {'seg': vols, 'post': vols_post}

            self._save_vols(sbj_vols, vols_filepath, **kwargs)

            remove_dir(tmp_dir)
            if target_tp == timepoints[-1]:
                print('.')
            else:
                print(',', end='', flush=True)

        print('DONE\n')


class USLR_LongSegment(LongLabelFusionProcessor, LongitudinalProcessor):
    """Label-fusion longitudinal segmentation using USLR affine displacements.

    Inherits from both :class:`~nicgiprep.processing.LongLabelFusionProcessing`
    (label-fusion logic) and :class:`~nicgiprep.processing.USLRProcessing`
    (BIDS entity templates). The displacement used to warp each atlas to the
    target space is derived from the linear USLR affine graph.
    """

    @property
    def pipeline_dir(self):
        """Output pipeline directory key."""
        return self._pipeline_dir

    @pipeline_dir.setter
    def pipeline_dir(self, value):
        """Set the output pipeline directory key."""
        self._pipeline_dir = value

    def _name(self):
        """Return the display name of this pipeline."""
        return 'USLRLabelFusion (gaussian kernel)'

    def _build_processor(self):
        """Extend the base processor with a pipeline-specific temp directory."""
        super()._build_processor()
        self.tmp_dir = join(self.tmp_dir, 'USLR-LF-' + str(self.pipeline_dir))
        create_dir(self.tmp_dir)

    def _get_tp_displacement(self, subject, target_tp, atlas_tp, **kwargs):
        """Build a composite displacement (affine + zero-field) from atlas to target.

        Composes the USLR affine transforms to produce a displacement field
        that maps the atlas timepoint to the target timepoint's segmentation
        space.

        Parameters
        ----------
        subject : str
            Subject ID.
        target_tp : str
            Session ID of the target (reference) timepoint.
        atlas_tp : str
            Session ID of the atlas (moving) timepoint.
        **kwargs
            Ignored.

        Returns
        -------
        np.ndarray or None
            Dense displacement field in the target space, or ``None`` if any
            required file is missing.
        """
        seg_ref_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': target_tp})
        aff_ref_file = self._get_data(**{'subject': subject, 'session': target_tp, **self.aff_graph_entities})
        if seg_ref_file is None or aff_ref_file is None:
            return

        proxysegref = nib.load(seg_ref_file.path)
        affref = np.linalg.inv(np.load(aff_ref_file.path))

        seg_flo_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': atlas_tp})
        aff_flo_file = self._get_data(**{'subject': subject, 'session': atlas_tp, **self.aff_graph_entities})
        if seg_flo_file is None or aff_flo_file is None:
            return

        proxysegflo = nib.load(seg_flo_file.path)
        affflo = np.load(aff_flo_file.path)

        flow = np.zeros(self.net_shape + (3,))
        return compose_transforms((tf.cast(np.linalg.inv(proxysegflo.affine) @ affflo, tf.float32), flow,
                                   tf.cast(affref @ proxysegref.affine, tf.float32)),
                                  shift_center=False, shape=proxysegref.shape)


    # def _deform_atlases(self, subject, target_tp, timepoints, results_dir, *args, **kwargs):
    #     seg_ref_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': target_tp})
    #     if seg_ref_file is None:
    #         return
    #
    #     proxysegref = nib.load(seg_ref_file.path)
    #     for atlas_tp in timepoints:
    #         im_file = self._get_data(**{**self.bf_entities, 'subject': subject, 'session': atlas_tp})
    #         seg_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': atlas_tp})
    #
    #         if im_file is None or seg_file is None:
    #             continue
    #
    #         output_image_filepath = join(results_dir, str(atlas_tp) + '.im.nii.gz')
    #         output_onehot_filepath = join(results_dir, str(atlas_tp) + '.onehot.nii.gz')
    #
    #         # One-hot encoding of the labels
    #         proxyseg = nib.load(seg_file.path)
    #         seg_arr = np.array(proxyseg.dataobj)
    #         onehot_arr = one_hot_encoding(seg_arr, categories=self.labels_lut)
    #
    #         # Gaussian filter for [1, 1, 1] segmentation
    #         proxyim = nib.load(im_file.path)
    #         arrayim = np.array(proxyim.dataobj)
    #         arrayim = gaussian_antialiasing(arrayim, proxyim.affine, [1, 1, 1])
    #         proxyim = nib.Nifti1Image(arrayim, proxyim.affine)
    #         proxyim = vol_resample_fast(proxyseg, proxyim)
    #         arrayim = np.array(proxyim.dataobj)
    #
    #         if target_tp == atlas_tp:
    #             proxyonehot = nib.Nifti1Image(onehot_arr.astype('float32'), proxyseg.affine)
    #             nib.save(proxyonehot, output_onehot_filepath)
    #             nib.save(proxyim, output_image_filepath)
    #
    #         else:
    #             tp_displ = self._get_tp_displacement(subject, target_tp, atlas_tp, **kwargs)
    #
    #             # Image
    #             arrayim_mov = warp(arrayim, tp_displ)
    #             proxyim = nib.Nifti1Image(arrayim_mov.astype('float32'), proxysegref.affine)
    #             nib.save(proxyim, output_image_filepath)
    #
    #             arrayonehot_mov = warp(onehot_arr, tp_displ)
    #             proxyonehot = nib.Nifti1Image(arrayonehot_mov.astype('float32'), proxysegref.affine)
    #             nib.save(proxyonehot, output_onehot_filepath)

    def process_subject(self, subject, cost='bch-l2', time_scale=None, g_std=None, **kwargs):
        """Run USLR label-fusion segmentation for one subject.

        Parameters
        ----------
        subject : str
            Subject ID.
        cost : {'uslr', 'uslr-lin'}, optional
            Which USLR derivative to use as displacement source. Default is
            ``'bch-l2'`` (treated as ``'uslr'``).
        time_scale : float, optional
            Temporal kernel scale. ``None`` disables temporal weighting.
        g_std : float, optional
            Appearance kernel standard deviation. ``None`` disables
            appearance weighting.
        **kwargs
            Forwarded to
            :meth:`~nicgiprep.processing.LongLabelFusionProcessing.process_subject`.
        """
        assert cost in ['uslr', 'uslr-lin']
        if cost not in ['uslr-lin']:
            self.svf_graph_entities['scope'] = 'uslr'

        super(LongLabelFusionProcessor, self).process_subject(subject, time_scale=time_scale, g_std=g_std, **kwargs)
