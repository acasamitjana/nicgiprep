

from os.path import isfile, join, dirname, basename, exists, isdir
from os import makedirs, listdir
import warnings

import pandas as pd
import nibabel as nib
from skimage.morphology import ball, binary_dilation

from setup import *
from nicgiprep.callbacks import *
from nicgiprep.pipelines import Processor
from nicgiprep.utils.preprocessing_utils import *
from nicgiprep.utils.fn_utils import one_hot_encoding, rescale_voxel_size
from nicgiprep.utils.label_utils import SYNTHSEG_LUT, CSF_LABELS, CLUSTER_DICT
from nicgiprep.utils.def_utils import vol_resample_fast
from nicgiprep.utils.io_utils import save_nii

class CrossSectionalProcessor(Processor):
    """Base class for cross-sectional neuroimaging processing pipelines.

    Thin wrapper around :class:`~nicgiprep.pipelines.base.Processor` that
    initialises common cross-sectional pipeline state via
    :meth:`_build_processor`.
    """

    def _build_processor(self, **kwargs):
        """Builds cross-sectional pipeline, by initialising relevant variables.

        Subclasses should call ``super()._build_processor()``
        and then extend or override these attributes.
        """

        super()._build_processor(**kwargs)

class SynthSegProcessor(CrossSectionalProcessor):
    """Run SynthSeg parcellation on all T1w images in a BIDS dataset.

    Collects input/output paths across subjects and sessions, invokes
    ``mri_synthseg`` in a single batch call, and aggregates the resulting
    volume TSV files into per-subject CSVs.
    """

    def _name(self):
        """Return the display name of this pipeline."""
        return 'SynthSegSegmentation'

    def _check_file(self, proxy):
        """Check whether a NIfTI image is suitable for SynthSeg.

        Parameters
        ----------
        proxy : nibabel.Nifti1Image
            Loaded image proxy.

        Returns
        -------
        dict
            ``{'run_flag': bool, 'exit_message': str}`` — ``run_flag`` is
            ``True`` when the image passes all checks.
        """
        try:
            if len(proxy.shape) != 3:
                return {'run_flag': False, 'exit_message': 'File excluded due to wrong image dimensions'}


            elif any([s < 20 for s in proxy.shape]):
                return {'run_flag': False, 'exit_message': 'File excluded due to wrong image dimensions'}

            elif any([r > 7 for r in np.sum(np.sqrt(np.abs(proxy.affine * proxy.affine)), axis=0)[:3].tolist()]):
                return {'run_flag': False,
                        'exit_message': 'File excluded due to large resolution in some image dimension.'}

            else:
                return {'run_flag': True, 'exit_message': ''}

        except:
            return {'run_flag': False,
                    'exit_message': 'File excluded due to an error reading the file or computing image shape and resolution.'}

    def _select_image(self, subject, tp):
        """Select a single T1w image for a given subject and session.

        Parameters
        ----------
        subject : str
            Subject ID.
        tp : str
            Session ID.

        Returns
        -------
        BIDSFile or None
            The selected T1w file, or ``None`` if none is found.
        """
        # Select a single T1w image per session
        t1w_list = self._get_data(subject=subject, extension=['nii', 'nii.gz'], suffix='T1w', session=tp, acquisition=['orig', None], scope='raw', ignore_check=True)

        if len(t1w_list) == 0:
            return None

        elif len(t1w_list) > 1:
            if any(['acquisition' not in f.entities.keys() for f in t1w_list]):
                t1w_list_r = list(filter(lambda x: 'acquisition' not in x.entities.keys(), t1w_list))

            elif any(['run' in f.entities.keys() for f in t1w_list]):
                t1w_list_r = list(filter(lambda x: x.entities['run'] == '01', t1w_list))

            else:
                t1w_list_r = t1w_list

            t1w_i = t1w_list_r[0]

        else:
            t1w_i = t1w_list[0]

        return t1w_i

    def process_parallel(self, num_cores, **kwargs):
        """Sequential fallback — SynthSeg cannot be parallelised.

        Emits a warning and delegates to :meth:`process`.
        """
        warnings.warn('Parallel implementation not possible for SynthSeg segmentation. It defers to sequential '
                      'processing')

        return self.process(**kwargs)

    def process(self, prefix='', gpu_flag=False, threads=16, **kwargs):
        """Collect all input/output paths and invoke SynthSeg in one batch call.

        Parameters
        ----------
        prefix : str, optional
            Prefix for temporary file lists written to ``TMP_DIR``. Default
            is ``''``.
        gpu_flag : bool, optional
            If ``True``, enable GPU inference. Default is ``False``.
        threads : int, optional
            Number of CPU threads for SynthSeg. Default is 16.
        **kwargs
            Forwarded to :meth:`process_subject`.
        """
        self._on_pipeline_init()

        input_files, res_files, output_files, vol_files, discarded_files = [], [], [], [], []
        for subject in self.subject_list:
            output = self.process_subject(subject, **kwargs)
            input_files.extend(output[0])
            res_files.extend(output[1])
            output_files.extend(output[2])
            vol_files.extend(output[3])
            discarded_files.extend(output[4])

        with open(join(TMP_DIR, prefix + '_input_files.txt'), 'w') as f:
            for i_f in input_files:
                f.write(i_f)
                f.write('\n')

        with open(join(TMP_DIR, prefix + '_res_files.txt'), 'w') as f:
            for i_f in res_files:
                f.write(i_f)
                f.write('\n')

        with open(join(TMP_DIR, prefix + '_output_files.txt'), 'w') as f:
            for i_f in output_files:
                f.write(i_f)
                f.write('\n')

        with open(join(TMP_DIR, prefix + '_vol_files.txt'), 'w') as f:
            for i_f in vol_files:
                f.write(i_f)
                f.write('\n')

        if len(output_files) >= 1:
            gpu_cmd = [''] if gpu_flag else ['--cpu']
            subprocess.call(['mri_synthseg',
                             '--i', join(TMP_DIR, prefix + '_input_files.txt'),
                             '--o', join(TMP_DIR, prefix + '_output_files.txt'),
                             '--resample', join(TMP_DIR, prefix + '_res_files.txt'),
                             '--vol', join(TMP_DIR, prefix + '_vol_files.txt'),
                             '--threads', str(threads), '--robust', '--parc'] + gpu_cmd)

        df_new = None
        for subject in listdir(DIR_PIPELINES['preproc']):
            if not isdir(join(DIR_PIPELINES['preproc'], subject)):
                continue

            for sess in listdir(join(DIR_PIPELINES['preproc'], subject)):
                if not exists(join(DIR_PIPELINES['preproc'], subject, sess, 'anat')): continue
                files = list(filter(lambda x: 'T1wdseg.csv' in x or 'T1wdseg.tsv' in x,
                                    listdir(join(DIR_PIPELINES['preproc'], subject, sess, 'anat'))))
                for f in files:
                    df = pd.read_csv(join(DIR_PIPELINES['preproc'], subject, sess, 'anat', f), dtype=str)
                    if len(df.columns) == 1:
                        df = pd.read_csv(join(DIR_PIPELINES['preproc'], subject, sess, 'anat', f), sep='\t', dtype=str)

                    if df_new is None:
                        df_new = pd.DataFrame(columns=['session'] + [c for c in df.columns if 'Unnamed' not in c])

                    if 'session' not in df.columns:
                        for _, row in df.iterrows():
                            try:
                                sid = row['Unnamed: 0'].split('ses-')[-1].split('_')[0]
                                row.drop('Unnamed: 0', inplace=True)
                                row['session'] = sid
                                df_new = pd.concat([df_new, row.to_frame().T])
                            except:
                                pass

                        df_new.set_index('session', drop=False, inplace=True)
                    else:
                        df_new = pd.concat([df_new, df])

            if df_new is not None:
                df_new.to_csv(join(DIR_PIPELINES['preproc'], subject, subject + '_vols.csv'), index=False)
            df_new = None

        self._update_full_layout()

    def process_subject(self, subject, force_flag=False, check_seg=None, **kwargs):
        """Collect SynthSeg I/O paths for all sessions of one subject.

        Parameters
        ----------
        subject : str
            Subject ID.
        force_flag : bool, optional
            If ``True``, re-queue sessions even when segmentations exist.
            Default is ``False``.
        check_seg : str, optional
            Directory to copy pre-existing segmentations from. Defaults to
            ``'/'`` (no pre-existing results).

        Returns
        -------
        tuple of list
            ``(input_files, res_files, output_files, vol_files,
            discarded_files)`` — one entry per session queued.
        """
        if check_seg is None:
            check_seg = '/'

        input_files, res_files, output_files, vol_files, discarded_files = [], [], [], [], []

        timepoints = self.bids_loader.get_session(subject=subject)
        for tp in timepoints:
            # Check if segmentation already exists
            preproc_dirname = join(DIR_PIPELINES['preproc'], 'sub-' + subject, 'ses-' + tp, 'anat')

            t1w_file = self._select_image(subject, tp)
            if t1w_file is None:
                continue

            if not exists(preproc_dirname): os.makedirs(preproc_dirname)
            f = open(join(preproc_dirname, t1w_file.filename.replace('nii.gz', 'txt')), 'w')
            f.write(
                'Since there exists more than one T1w image for this session, we choose this file to run over the '
                'entire USLR pipeline with the corresponding segmentation. Refer to the rawdata to check '
                'correspondence'
            )

            raw_dirname = t1w_file.dirname
            t1w_entities = {k: str(v) for k, v in t1w_file.entities.items() if k in filename_entities}
            t1w_entities['acquisition'] = '1'

            anat_res = basename(self.build_path(t1w_entities))
            anat_seg = anat_res.replace('T1w', 'T1wdseg')
            anat_vols = anat_seg.replace('nii.gz', 'tsv')

            if (exists(join(check_seg, 'sub-' + subject, 'ses-' + tp, 'anat', anat_seg)) and
                    exists(join(check_seg, 'sub-' + subject, 'ses-' + tp, 'anat', anat_vols))):
                subprocess.call(['cp', join(check_seg, 'sub-' + subject, 'ses-' + tp, 'anat', anat_seg), join(preproc_dirname, anat_seg)])
                subprocess.call(['cp', join(check_seg, 'sub-' + subject, 'ses-' + tp, 'anat', anat_vols), join(preproc_dirname, anat_vols)])


            if not exists(join(preproc_dirname, anat_seg)) or force_flag:
                proxy = nib.load(join(raw_dirname, t1w_file.filename))
                run_code = self._check_file(proxy)
                if run_code['run_flag']:
                    input_files += [join(raw_dirname, t1w_file.filename)]
                    res_files += [join(preproc_dirname, anat_res)]
                    output_files += [join(preproc_dirname, anat_seg)]
                    vol_files += [join(preproc_dirname, anat_vols)]
                else:
                    with open(join(preproc_dirname, 'excluded_file.txt'), 'w') as f:
                        f.write(run_code['exit_message'])

        return input_files, res_files, output_files, vol_files, discarded_files


class BiasCorrectionProcessor(CrossSectionalProcessor):
    """Bias-field correction and brain-mask generation for cross-sectional data.

    For each session, computes a brain mask from the SynthSeg parcellation,
    resamples the T1w to 1 mm isotropic if needed, runs EM-based bias-field
    correction, and normalises the white-matter mean to 110.
    """

    def _name(self):
        """Return the display name of this pipeline."""
        return 'BiasFieldCorrection'

    def _check_resampled_file(self, raw_file, resampled_entities):
        """Ensure a 1 mm isotropic resampled T1w exists, creating it if necessary.

        Parameters
        ----------
        raw_file : BIDSFile
            Source raw T1w file.
        resampled_entities : dict
            BIDS entities for the target 1 mm resampled file.

        Returns
        -------
        dict
            ``{'exit_code': int, ...}`` — ``exit_code=0`` with ``'filepath'``
            on success; ``exit_code=-1`` with ``'message'`` on failure.
        """
        resampled_file = self._get_data(**resampled_entities, ignore_check=True)
        if not resampled_file:
            resampled_filepath = join(DIR_PIPELINES['preproc'], self.build_path(resampled_entities))

            proxyraw = nib.load(raw_file.path)
            pixdim = np.sqrt(np.sum(proxyraw.affine * proxyraw.affine, axis=0))[:-1]
            if all([np.abs(p - 1) < 0.01 for p in pixdim]):
                rf = subprocess.call(['ln', '-s', raw_file.path, resampled_filepath], stderr=subprocess.PIPE)
                if rf != 0:
                    subprocess.call(['cp', raw_file.path, resampled_filepath])

            else:
                # some dimension may be wrong
                if any([p < 0.01 for p in pixdim]):
                    return {'exit_code': -1, 'message': 'some dimensions are wrong'}

                v, aff = rescale_voxel_size(np.array(proxyraw.dataobj), proxyraw.affine, [1, 1, 1])
                save_nii(v, aff, resampled_filepath)

        else:
            resampled_filepath = resampled_file[0].path

        return {'exit_code': 0, 'filepath': resampled_filepath}

    def _posterior2generative_labelmap(self, seg, lut=SYNTHSEG_LUT):
        """Convert soft posterior segmentation to a hemisphere-unified label space.

        Merges bilateral structures into unified classes using ``CLUSTER_DICT`` and
        re-normalises the resulting posteriors so they sum to one across classes.

        Parameters
        ----------
        seg : np.ndarray
            Soft segmentation array of shape ``(*spatial_dims, n_labels)``, where
            the last axis indexes labels according to ``lut``.
        lut : dict, optional
            Lookup table mapping FreeSurfer/SynthSeg integer label IDs to channel
            indices in ``seg``. Defaults to ``SYNTHSEG_LUT``.

        Returns
        -------
        np.ndarray
            Array of shape ``(*spatial_dims, n_unified_classes)`` with normalised
            posterior probabilities over the unified label set.
        """
        out_seg = np.zeros(seg.shape[:-1] + (len(CLUSTER_DICT.keys()),))
        for it_lab, (lab_str, lab_list) in enumerate(CLUSTER_DICT.items()):
            for lab in lab_list:
                out_seg[..., it_lab] += seg[..., lut[lab]]

        out_seg = out_seg / (np.sum(out_seg, axis=-1, keepdims=True) + 1e-10)
        out_seg[np.isnan(out_seg)] = 0
        return out_seg

    def _seg2generative_labelmap(self, seg):
        """Convert a hard integer segmentation to a hemisphere-unified label space.

        Maps bilateral FreeSurfer/SynthSeg integer labels to unified class indices
        defined by ``CLUSTER_DICT``.

        Parameters
        ----------
        seg : np.ndarray
            Integer segmentation array of shape ``(H, W, D)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(H, W, D, n_unified_classes)`` with class assignments
            normalised across the last axis.

        Notes
        -----
        Division by the sum along the last axis may produce NaN at background
        voxels where the total is zero; callers should handle this if needed.
        """
        out_seg = np.zeros(seg.shape[:3] + (len(CLUSTER_DICT.keys()),))
        for it_lab, (lab_str, lab_list) in enumerate(CLUSTER_DICT.items()):
            for lab in lab_list:
                out_seg[seg == lab] = it_lab

        out_seg = out_seg / np.sum(out_seg, axis=-1, keepdims=True)

        return out_seg


    def process_subject(self, subject, force_flag=False, remove_wrong=True, **kwargs):
        """Run bias-field correction for all sessions of one subject.

        Parameters
        ----------
        subject : str
            Subject ID.
        force_flag : bool, optional
            If ``True``, reprocess sessions even when outputs already exist.
            Default is ``False``.
        remove_wrong : bool, optional
            If ``True``, delete the session directory when bias-field
            correction fails. Default is ``True``.
        **kwargs
            Ignored; present for API compatibility.
        """
        print('\nSubject: ' + subject)

        timepoints = self._get_timepoints(subject=subject, uslr=True)
        for tp_id in timepoints:
            print('\n* Session: ' + tp_id, end=': ', flush=True)

            preproc_dirname = join(DIR_PIPELINES['preproc'], 'sub-' + subject, 'ses-' + tp_id, 'anat')
            if not exists(preproc_dirname): os.makedirs(preproc_dirname)

            # input segs
            seg_file = self._get_data(**{'session': tp_id, 'subject': subject, **self.seg_entities})
            if seg_file is None:
                continue

            # get entities
            seg_entities = self._get_entities(seg_file)
            seg_entities['extension'] = 'nii.gz'
            raw_entities = {k: str(v) for k, v in seg_entities.items() if k != 'acquisition'}
            raw_entities['suffix'] = 'T1w'
            raw_entities['scope'] = 'raw'
            raw_entities['acquisition'] = [None, 'orig']
            resampled_entities = copy.copy(raw_entities)
            resampled_entities['acquisition'] = '1'
            resampled_entities['scope'] = 'preproc'

            # raw image
            raw_file = self._get_data(**raw_entities)
            if raw_file is None:
                continue

            # build output paths
            output_filepath = join(preproc_dirname, basename(raw_file))
            output_mask_filepath = join(preproc_dirname, seg_file.filename.replace('dseg', 'mask'))

            if exists(output_filepath) and exists(output_mask_filepath) and not force_flag:
                print('image already processed.')
                continue

            # read images
            proxyraw = nib.load(raw_file.path)
            proxyseg = nib.load(seg_file.path)

            # ------------------------ #
            #      Computing masks     #
            # ------------------------ #
            # print('computing masks from dseg files; ', end='', flush=True)
            if not exists(output_mask_filepath):
                seg = np.array(proxyseg.dataobj)
                mask = seg > 0
                for lab in CSF_LABELS:
                    mask[seg == lab] = 0

                save_nii(mask.astype('uint8'), proxyseg.affine, output_mask_filepath)

            resampled_flag = self._check_resampled_file(raw_file, resampled_entities)
            if resampled_flag['exit_code'] == -1:
                print(resampled_flag['message'], end='', flush=True)
                continue

            proxyres = nib.load(resampled_flag['filepath'])

            # ------------------------ #
            # Bias field correction    #
            # ------------------------ #
            # print('correcting for inhomogeneities and normalisation (min/max); ', end='', flush=True)
            if not exists(output_filepath) or force_flag:
                vox2ras0 = proxyres.affine
                mri_acq = np.asarray(proxyres.dataobj)
                mri_acq[np.isnan(mri_acq)] = 0

                pixdimim = np.sqrt(np.sum(proxyres.affine * proxyres.affine, axis=0))[:-1]
                pixdimseg = np.sqrt(np.sum(proxyseg.affine * proxyseg.affine, axis=0))[:-1]
                if any([np.abs(p1-p2)> 0.01 for p1, p2 in zip(pixdimseg, pixdimim)]):
                    proxyseg = vol_resample_fast(proxyres, proxyseg, mode='nearest')

                seg = np.array(proxyseg.dataobj)
                soft_seg = one_hot_encoding(seg, categories=SYNTHSEG_LUT)
                soft_seg = self._posterior2generative_labelmap(soft_seg, lut=SYNTHSEG_LUT)
                try:
                    mri_acq_corr, bias_field = bias_field_corr(mri_acq, soft_seg, penalty=1, VERBOSE=False, filter_exceptions=True)
                except:
                    mri_acq_corr = None

                if mri_acq_corr is None:
                    if not remove_wrong:
                        print("[error] bias field cannot be computed -- removing segmentation related files and "
                              "exiting: " + seg_file.path, end='\n')
                        subprocess.call(['rm', '-rf', join(DIR_PIPELINES['preproc'], 'sub-' + subject, 'ses-' + tp_id)])

                    else:
                        print("[error] bias field cannot be computed.", end='', flush=True)

                    continue

                del soft_seg

                mask = seg > 0
                wm_mask = (seg == 2) | (seg == 41)

                del seg

                vox2ras0_orig = proxyraw.affine
                mri_acq_orig = np.asarray(proxyraw.dataobj)
                mri_acq_orig[np.isnan(mri_acq_orig)] = 0
                if len(mri_acq_orig.shape) > 3:
                    mri_acq_orig = mri_acq_orig[..., 0]

                new_vox_size = np.linalg.norm(vox2ras0_orig, 2, 0)[:3]
                vox_size = np.linalg.norm(vox2ras0, 2, 0)[:3]

                if all([v1 == v2 for v1, v2 in zip(vox_size, new_vox_size)]):
                    mask_dilated = binary_dilation(mask, ball(3))
                    m = np.mean(mri_acq_corr[wm_mask])
                    mri_acq_corr = 110 * mri_acq_corr / m
                    mri_acq_corr *= mask_dilated

                    save_nii(np.clip(mri_acq_corr, 0, 255).astype('uint8'), proxyres.affine, output_filepath)

                else:
                    bias_proxy = nib.Nifti1Image(bias_field, proxyres.affine)
                    bias_field_resize = vol_resample_fast(proxyraw, bias_proxy, return_np=True)
                    #
                    mask_proxy = nib.Nifti1Image(mask.astype('float'), proxyres.affine)
                    mask_resize = vol_resample_fast(proxyraw, mask_proxy, return_np=True) > 0.5
                    #
                    wm_mask_proxy = nib.Nifti1Image(wm_mask.astype('float'), proxyres.affine)
                    wm_mask_resize = vol_resample_fast(proxyraw, wm_mask_proxy, return_np=True) > 0.5

                    mri_acq_orig_corr = copy.copy(mri_acq_orig.astype('float32'))
                    mri_acq_orig_corr[mask_resize] = mri_acq_orig_corr[mask_resize] / bias_field_resize[mask_resize]

                    m = np.mean(mri_acq_orig_corr[wm_mask_resize])
                    mri_acq_orig_corr = 110 * mri_acq_orig_corr / m
                    mask_dilated = binary_dilation(mask_resize, ball(3))
                    mri_acq_orig_corr[mask_dilated == 0] = 0

                    save_nii(np.clip(mri_acq_orig_corr, 0, 255).astype('uint8'), proxyraw.affine, output_filepath)

                    del bias_field, bias_field_resize, mri_acq_orig, mri_acq_orig_corr, mask_dilated

        return {'exit_code': 0, 'message': 'success'}


