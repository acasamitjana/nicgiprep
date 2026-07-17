"""
Processing pipeline classes for cross-sectional T1w MRI data in BIDS format.

This involves running SuperSynth and SynthSeg to get label maps, followed by
Bias Field correction and MNI registration.

"""
import pdb
from os import listdir
from os.path import join, dirname, basename, exists, isdir
from typing import List, Optional, Tuple, Dict
import warnings
import subprocess

import tqdm
from bids.layout import BIDSFile, parse_file_entities
import pandas as pd
import nibabel as nib
from skimage.morphology import ball, binary_dilation

from setup import *
from nicgiprep.pipelines.base import Processor
from nicgiprep.utils.preprocessing_utils import *
from nicgiprep.utils.fn_utils import one_hot_encoding, rescale_voxel_size
from nicgiprep.utils.io_utils import save_volume, remove_duplicates_csv
from nicgiprep.utils.def_utils import vol_resample_fast, register_to_MNI
from nicgiprep.utils.label_utils import SYNTHSEG_LUT, CSF_LABELS, SYNTHSEG_GMM_ONTOLOGY, labels_registration


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


class MRISegmentationProcessor(CrossSectionalProcessor):
    """Run SuperSynth and SynthSeg parcellation on all T1w images in a BIDS dataset.

    Collects input/output paths across subjects and sessions, invokes
    ``mri_super_synth`` and ``mri_synthseg`` in a single batch call, and aggregates the resulting
    volumes and qc files into per-subject TSV files.
    """

    def _name(self):
        """Return the display name of this pipeline."""
        return "SuperSynthSegSegmentation"

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
                return {
                    "run_flag": False,
                    "exit_message": "File excluded due to wrong image dimensions",
                }

            elif any([s < 20 for s in proxy.shape]):
                return {
                    "run_flag": False,
                    "exit_message": "File excluded due to wrong image dimensions",
                }

            elif any(
                [
                    r > 7
                    for r in np.sum(
                        np.sqrt(np.abs(proxy.affine * proxy.affine)), axis=0
                    )[:3].tolist()
                ]
            ):
                return {
                    "run_flag": False,
                    "exit_message": "File excluded due to large resolution in some image dimension.",
                }

            else:
                return {"run_flag": True, "exit_message": ""}

        except:
            return {
                "run_flag": False,
                "exit_message": "File excluded due to an error reading the file or computing image shape and resolution.",
            }

    def _select_images(self, subject: str, sess: str, num: Optional[int] = None,) -> Optional[BIDSFile|List[BIDSFile]]:
        """Select a single T1w image for a given subject and session.

        Parameters
        ----------
        subject : str
            Subject ID.
        sess : str
            Session ID.
        num : int, optional
            Number of retrieved images. If None, all images are returned, otherwise, we randomly select min(image, num)

        Returns
        -------
        BIDSFile or None
            The selected T1w file, or ``None`` if none is found.
        """
        # Select a single T1w image per session
        t1w_list = self._get_data(
            subject=subject,
            session=sess,
            suffix="T1w",
            extension=[".nii", ".nii.gz"],
            acquisition=["orig", None],
            scope="raw",
            ignore_check=True,
        )

        if num is None:
            return t1w_list

        if len(t1w_list) == 0:
            return None

        elif len(t1w_list) > 1:
            if any(["acquisition" not in f.entities.keys() for f in t1w_list]):
                t1w_list_r = list(
                    filter(lambda x: "acquisition" not in x.entities.keys(), t1w_list)
                )
            elif any(["run" in f.entities.keys() for f in t1w_list]):
                t1w_list_r = list(filter(lambda x: x.entities["run"] == "01", t1w_list))
            else:
                t1w_list_r = t1w_list
            t1w_i = t1w_list_r[0]

        else:
            t1w_i = t1w_list[0]

        return t1w_i

    def process_parallel(self, num_cores, **kwargs):  # TODO: Fix this or remove it
        """Sequential fallback — SynthSeg cannot be parallelised.

        Emits a warning and delegates to :meth:`process`.
        """
        warnings.warn(
            "Parallel implementation not possible for SynthSeg segmentation. It defers to sequential "
            "processing"
        )

        return self.process(**kwargs)

    def process(
        self, prefix: str = "", gpu_flag: bool = False, threads: int = 16, **kwargs
    ):
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

        # self.bids_loader.get(**{"session": sess_id, "subject": subject, 'suffix': ['T1wsynthseg', 'synthseg']})
        csv_file = join(TMP_DIR, "files_to_process_supersynth.csv")  ## CSV file for SuperSynth batch processing
        if exists(csv_file):
            os.remove(csv_file)

        input_files, res_files, synthseg_files, supersynth_files = (
            [],
            [],
            [],
            {'subject': [], 'session': [], 'input_file': [], 'output_fname': [], 'mode': [], 'tmp_dir': [], 'run': []},
        )
        for subject in self.subject_list:
            output = self.process_subject(subject, **kwargs)
            input_files.extend(output[0])
            res_files.extend(output[1])
            synthseg_files.extend(output[2])
            supersynth_files['subject'].extend(output[3]['subject'])
            supersynth_files['session'].extend(output[3]['session'])
            supersynth_files['input_file'].extend(output[3]['input_file'])
            supersynth_files['output_fname'].extend(output[3]['output_fname'])
            supersynth_files['mode'].extend(output[3]['mode'])
            supersynth_files['tmp_dir'].extend(output[3]['tmp_dir'])
            supersynth_files['run'].extend(output[3]['run'])

        with open(join(TMP_DIR, prefix + "_input_files.txt"), "w") as f:
            for i_f in input_files:
                f.write(i_f)
                f.write("\n")

        with open(join(TMP_DIR, prefix + "_res_files.txt"), "w") as f:
            for i_f in res_files:
                f.write(i_f)
                f.write("\n")

        with open(join(TMP_DIR, prefix + "_synthseg_files.txt"), "w") as f:
            for i_f in synthseg_files:
                f.write(i_f)
                f.write("\n")

        supersynth_df = pd.DataFrame(supersynth_files)

        run_df = copy.deepcopy(supersynth_df)
        run_df = run_df[run_df['run']]

        if len(synthseg_files) >= 1:
            gpu_cmd = [] if gpu_flag else ["--cpu"]
            ## Run SynthSeg
            subprocess.call(
                [
                    "mri_synthseg",
                    "--i",
                    join(TMP_DIR, prefix + "_input_files.txt"),
                    "--o",
                    join(TMP_DIR, prefix + "_synthseg_files.txt"),
                    "--resample",
                    join(TMP_DIR, prefix + "_res_files.txt"),
                    "--threads",
                    str(threads),
                    "--robust",
                    "--parc",
                ]
                + gpu_cmd
            )

        if len(run_df) >= 1:
            run_df = run_df[['input_file', 'tmp_dir', 'mode']]
            run_df.to_csv(join(TMP_DIR, prefix + "_supersynth_files.csv"), header=False, index=False)

            subprocess.call(
                [
                    "mri_super_synth",
                    "--i",
                    join(TMP_DIR, prefix + "_supersynth_files.csv"),
                    "--threads",
                    str(threads),
                    "--device",
                    "cuda" if gpu_flag else "cpu",
                ]
            )

            ## Checking if some cases were skipped due to CUDA OOM
            run_df = pd.DataFrame()
            for _, row in supersynth_df.iterrows():
                if row['run'] and not exists(join(row['tmp_dir'], "segmentation.mgz")):
                    run_df = pd.concat([run_df, row.to_frame().T])

            if len(run_df) > 0:
                ## Running again SuperSynth using cpu
                run_df = run_df[['input_file', 'tmp_dir', 'mode']]
                run_df.to_csv(join(TMP_DIR, prefix + "_supersynth_files.csv"), header=False, index=False, mode='w')
                subprocess.call(
                    [
                        "mri_super_synth",  # ERROR: temporary fix using full path to supersynth
                        "--i",
                        join(TMP_DIR, prefix + "_supersynth_files.csv"),
                        "--threads",
                        str(threads),
                        "--device",
                        "cpu",
                    ]
                )

        # From all supersynth outputs, keep only the segmentation and resample it back to the original resolution.
        # Keep the 1x1x1 in the utils files
        supersynth_failed = []
        for _, row in supersynth_df.iterrows():
            subject_id, sess_id = row['subject'], row['session']
            input_file, tmp_dir =  row['input_file'], row['tmp_dir']

            proc_sess_dir = join(
                DIR_PIPELINES["nicgiprep-cross"],
                "sub-" + str(subject_id),
            )

            proc_anat_dir = join(
                proc_sess_dir,
                "ses-" + str(sess_id),
                "anat",
            )

            if not exists(proc_anat_dir):
                os.makedirs(proc_anat_dir)

            t1w_entities = self._get_entities(input_file)
            seg_entities = copy.deepcopy(t1w_entities)
            seg_entities['suffix'] += 'dseg'
            seg_fname = self.build_path(seg_entities)
            seg_entities['acquisition'] = '1'
            seg_entities['datatype'] = 'utils'
            seg_1x1x1_fname = self.build_path(seg_entities)
            vol_fname = 'sub-' + subject_id + '_summary.tsv'

            seg_file = join(tmp_dir, "segmentation.mgz")
            csv_file = join(tmp_dir, "volumes.csv")
            qc_file = join(tmp_dir, "qc.csv")


            if not exists(seg_file) or not exists(csv_file) or not exists(qc_file):
                #supersynth failed for whatever reason and continue
                supersynth_failed.append(input_file)
                continue

            # save original SuperSynth in the utils
            if not exists(join(DIR_PIPELINES["nicgiprep-cross"], seg_1x1x1_fname)):
                subprocess.call(['cp', seg_file, join(DIR_PIPELINES["nicgiprep-cross"], seg_1x1x1_fname)])

            # copy the volumes and qc data
            vols_df = pd.read_csv(csv_file, dtype=str)
            qc_df = pd.read_csv(qc_file, dtype=str)
            qc_df = qc_df.add_prefix('qc_')
            qc_df.columns.str.replace(" ", "_")

            summary_df = pd.concat([vols_df, qc_df], axis=1)
            summary_df.insert(loc=0, column='session', value=sess_id)

            if exists(join(proc_sess_dir, vol_fname)):
                existing_summary_df = pd.read_csv(join(proc_sess_dir, vol_fname), dtype=str, sep='\t')
                summary_df  = pd.concat([existing_summary_df, summary_df], axis=0)

            summary_df.drop_duplicates(inplace=True)
            summary_df.to_csv(join(proc_sess_dir, vol_fname), sep='\t', index=False)

            # read data
            im_proxy = nib.load(input_file)
            seg_proxy = nib.load(seg_file)
            seg_arr = np.array(seg_proxy.dataobj)

            # resample labelmaps to original resolution
            onehot_arr, onehot_lut = one_hot_encoding(seg_arr, return_lut=True)
            onehot_proxy = nib.Nifti1Image(
                onehot_arr.astype('float'), seg_proxy.affine
            )
            onehot_proxy = vol_resample_fast(
                im_proxy, onehot_proxy,
            )

            onehot_res_arr = np.array(onehot_proxy.dataobj)
            seg_argmax_res_arr = np.argmax(onehot_res_arr, axis=-1)
            seg_res_arr = np.zeros_like(seg_argmax_res_arr, dtype='int32')
            for ul, it_ul in onehot_lut.items():
                seg_res_arr[seg_argmax_res_arr == it_ul] = ul

            # save anat and utils files
            save_volume(seg_res_arr, aff=onehot_proxy.affine, path=join(DIR_PIPELINES['nicgiprep-cross'], seg_fname))

            # remove supersynth directory
            # subprocess.call(['rm', '-rf', tmp_dir])

        self._update_full_layout()

        print('\n')
        print("=" * 40)
        print("-" * 15 + "  SUMMARY  " + "-" * 14)
        print("=" * 40)
        print(f"Total Number of images processed: {len(supersynth_df)}")
        print(f"SUCCESS:                  {len(supersynth_df) - len(supersynth_failed)}/{len(supersynth_df)}")
        print(f"FAILED:                   {len(supersynth_failed)}/{len(supersynth_df)}")
        print("-" * 40)


    def process_subject(
        self,
        subject: str,
        force_flag: bool = False,
        session_list: Optional[List[str]] = None,
        check_seg: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List, List, List, Dict]:
        """Collect SynthSeg I/O paths for all sessions of one subject.

        Parameters
        ----------
        subject : str
            Subject ID.
        session_list : list of str, optional
            Restrict processing to these session IDs. If ``None``, all
            sessions for the subject are processed.
        force_flag : bool, optional
            If ``True``, re-queue sessions even when segmentations exist.
            Default is ``False``.
        check_seg : str, optional
            Directory to copy pre-existing segmentations from. Defaults to
            ``'/'`` (no pre-existing results).
        **kwargs
            Ignored; present for API compatibility.

        Returns
        -------
        tuple of list
            ``(input_files, res_files, output_files, vol_files,
            )`` — one entry per session queued.
        """
        input_files, res_files, output_files, supersynth_files = (
            [],
            [],
            [],
            {'subject': [], 'session': [], 'input_file': [], 'output_fname': [], 'mode': [], 'tmp_dir': [], 'run': []},
        )

        sessions = self.bids_loader.get_session(subject=subject)
        if session_list is not None:
            sessions = [s for s in sessions if s in session_list]

        for sess_id in sessions:
            t1w_files = self._select_images(subject, sess_id)
            if t1w_files is None:
                continue

            if isinstance(t1w_files, BIDSFile):
                t1w_files = [t1w_files]

            tmp_dir = join(
                TMP_DIR,
                "supersynth_process",
                "sub-" + subject,
                "ses-" + sess_id)

            proc_anat_dir = join(
                DIR_PIPELINES["nicgiprep-cross"],
                "sub-" + str(subject),
                "ses-" + str(sess_id),
                "anat",
            )

            proc_utils_dir = join(
                DIR_PIPELINES["nicgiprep-cross"],
                "sub-" + subject,
                "ses-" + sess_id,
                "utils",
            )

            for t1w_file in t1w_files:
                raw_dirname = t1w_file.dirname
                t1w_entities = dict(t1w_file.entities)
                t1w_entities["acquisition"] = "1"
                synthseg_entities = {**t1w_entities, "suffix": "T1wsynthseg"}

                anat_res = basename(self.build_path(t1w_entities))
                anat_seg = basename(self.build_path(synthseg_entities))

                # Check if segmentation already exists
                if (check_seg is not None and
                        exists(join(check_seg, "sub-" + subject, "ses-" + sess_id, "utils", anat_seg))):

                    subprocess.call(
                        [
                            "cp",
                            join(
                                check_seg,
                                "sub-" + subject,
                                "ses-" + sess_id,
                                "utils",
                                anat_seg,
                            ),
                            join(proc_utils_dir, anat_seg),
                        ]
                    )

                if not exists(join(proc_utils_dir, anat_seg)) or force_flag:

                    proxy = nib.load(join(raw_dirname, t1w_file.filename))
                    run_code = self._check_file(proxy)
                    if run_code["run_flag"]:
                        input_files += [join(raw_dirname, t1w_file.filename)]
                        res_files += [join(proc_utils_dir, anat_res)]
                        output_files += [join(proc_utils_dir, anat_seg)]

                    else:
                        with open(join(proc_utils_dir, "excluded_file.txt"), "w") as f:
                            f.write(run_code["exit_message"])

                seg_fname = t1w_file.filename.replace(".nii.gz", "dseg.nii.gz")

                if not exists(join(proc_anat_dir, seg_fname)):
                    os.makedirs(tmp_dir, exist_ok=True)
                    supersynth_files['subject'] += [subject]
                    supersynth_files['session'] += [sess_id]
                    supersynth_files['input_file'] += [f"{join(raw_dirname, t1w_file.filename)}"]
                    supersynth_files['tmp_dir'] += [tmp_dir]
                    supersynth_files['mode'] += ["invivo"]
                    supersynth_files['output_fname'] += [join(proc_anat_dir, seg_fname)]
                    if not exists(join(tmp_dir, "segmentation.mgz")):
                        supersynth_files['run'] += [True]
                    else:
                        supersynth_files['run'] += [False]

        return input_files, res_files, output_files, supersynth_files


class BiasCorrectionProcessor(CrossSectionalProcessor):
    """Bias-field correction and brain-mask generation for cross-sectional data.

    For each session, computes a brain mask from the SynthSeg parcellation,
    resamples the T1w to 1 mm isotropic if needed, runs EM-based bias-field
    correction, and normalises the white-matter mean to 110.
    """

    def _name(self):
        """Return the display name of this pipeline."""
        return "BiasFieldCorrection"

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
            resampled_filepath = join(
                DIR_PIPELINES["nicgiprep-cross"], self.build_path(resampled_entities)
            )
            proxyraw = nib.load(raw_file.path)
            pixdim = np.sqrt(np.sum(proxyraw.affine * proxyraw.affine, axis=0))[:-1]
            if all([np.abs(p - 1) < 0.01 for p in pixdim]):
                rf = subprocess.call(
                    ["ln", "-s", raw_file.path, resampled_filepath],
                    stderr=subprocess.PIPE,
                )
                if rf != 0:
                    subprocess.call(["cp", raw_file.path, resampled_filepath])

            else:
                # some dimension may be wrong
                if any([p < 0.01 for p in pixdim]):
                    return {"exit_code": -1, "message": "some dimensions are wrong"}

                v, aff = rescale_voxel_size(
                    np.array(proxyraw.dataobj), proxyraw.affine, [1, 1, 1]
                )
                save_volume(
                    volume=v,
                    aff=aff,
                    header=proxyraw.header,
                    path=resampled_filepath,
                    res=[1, 1, 1],
                )

        else:
            resampled_filepath = resampled_file[0].path

        return {"exit_code": 1, "filepath": resampled_filepath, "message": "success"}

    def _posterior2generative_labelmap(self, seg, lut=SYNTHSEG_LUT):
        """Convert soft posterior segmentation to a hemisphere-unified label space.

        Merges bilateral structures into unified classes using ``SYNTHSEG_GMM_ONTOLOGY`` and
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
        out_seg = np.zeros(seg.shape[:-1] + (len(SYNTHSEG_GMM_ONTOLOGY.keys()),))
        for it_lab, (lab_str, lab_list) in enumerate(SYNTHSEG_GMM_ONTOLOGY.items()):
            for lab in lab_list:
                out_seg[..., it_lab] += seg[..., lut[lab]]

        out_seg = out_seg / (np.sum(out_seg, axis=-1, keepdims=True) + 1e-10)
        out_seg[np.isnan(out_seg)] = 0
        return out_seg

    def _seg2generative_labelmap(self, seg):
        """Convert a hard integer segmentation to a hemisphere-unified label space.

        Maps bilateral FreeSurfer/SynthSeg integer labels to unified class indices
        defined by ``SYNTHSEG_GMM_ONTOLOGY``.

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
        out_seg = np.zeros(seg.shape[:3] + (len(SYNTHSEG_GMM_ONTOLOGY.keys()),))
        for it_lab, (lab_str, lab_list) in enumerate(SYNTHSEG_GMM_ONTOLOGY.items()):
            for lab in lab_list:
                out_seg[seg == lab] = it_lab

        out_seg = out_seg / np.sum(out_seg, axis=-1, keepdims=True)

        return out_seg

    def process_image(self, seg_file: BIDSFile, force_flag: bool=False):
        """Run bias-field correction for a given image.

        Parameters
        ----------
        seg_file : BIDSFile
            File to process. It should be a BIDSFile so that we can get the entities to build the outputs.
        force_flag : bool, optional
            If ``True``, reprocess sessions even when outputs already exist.
            Default is ``False``.

        Returns
        -------
        dict
            Result dict with keys:
              - ``exit_code`` (int): ``0`` on success / already processed, ``-1`` on failure.
              - ``message`` (str): human-readable description of the outcome.
        """

        subject = seg_file.entities['subject']
        sess_id = seg_file.entities['session']

        preproc_dirname = join(
            DIR_PIPELINES["nicgiprep-cross"],
            "sub-" + subject,
            "ses-" + sess_id,
            "utils",
        )

        # get entities
        seg_entities = self._get_entities(seg_file)
        seg_entities["extension"] = ".nii.gz"

        # build T1w rawdata entities
        raw_entities = copy.deepcopy(seg_entities)
        raw_entities["suffix"] = "T1w"
        raw_entities["scope"] = "raw"
        raw_entities["acquisition"] = [None, "orig"]
        raw_entities.pop("datatype", None)
        if 'run' not in raw_entities: raw_entities['run'] = None

        # build T1w corrected and resampled entities
        resampled_entities = copy.copy(raw_entities)
        resampled_entities["acquisition"] = "1"
        resampled_entities["scope"] = "nicgiprep-cross"
        resampled_entities["datatype"] = "utils"

        # raw image
        raw_file = self._get_data(**raw_entities)
        if raw_file is None:
            return

        # build output paths
        output_filepath = join(
            preproc_dirname.replace("utils", "anat"), basename(raw_file)
        )  ## Saving the final result in /anat
        output_mask_filepath = join(
            preproc_dirname, seg_file.filename.replace("synthseg", "mask")
        )

        if (
                exists(output_filepath)
                and exists(output_mask_filepath)
                and not force_flag
        ):
            return {
                "exit_code": 0,
                "message": "image already processed.",
            }

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

            save_volume(
                volume=mask.astype("uint8"),
                aff=proxyseg.affine,
                header=proxyseg.header,
                path=output_mask_filepath,
            )

        resampled_flag = self._check_resampled_file(raw_file, resampled_entities)
        if resampled_flag["exit_code"] == -1:
            return resampled_flag

        proxyres = nib.load(resampled_flag["filepath"])

        # ------------------------ #
        # Bias field correction    #
        # ------------------------ #
        # print('correcting for inhomogeneities and normalisation (min/max); ', end='', flush=True)
        if not exists(output_filepath) or force_flag:
            vox2ras0 = proxyres.affine
            mri_acq = np.asarray(proxyres.dataobj)
            mri_acq[np.isnan(mri_acq)] = 0

            pixdimim = np.sqrt(np.sum(proxyres.affine * proxyres.affine, axis=0))[
                :-1
            ]
            pixdimseg = np.sqrt(np.sum(proxyseg.affine * proxyseg.affine, axis=0))[
                :-1
            ]
            if any([np.abs(p1 - p2) > 0.01 for p1, p2 in zip(pixdimseg, pixdimim)]):
                proxyseg = vol_resample_fast(proxyres, proxyseg, mode="nearest")

            seg = np.array(proxyseg.dataobj)
            soft_seg = one_hot_encoding(seg, categories=SYNTHSEG_LUT)
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
            except:
                return {
                    "exit_code": -1,
                    "message": "bias field cannot be computed.",
                }

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

                save_volume(
                    volume=np.clip(mri_acq_corr, 0, 255).astype("uint8"),
                    aff=proxyres.affine,
                    header=proxyres.header,
                    path=output_filepath,
                )

            else:
                bias_proxy = nib.Nifti1Image(bias_field, proxyres.affine)
                bias_field_resize = vol_resample_fast(
                    proxyraw, bias_proxy, return_np=True
                )
                #
                mask_proxy = nib.Nifti1Image(mask.astype("float"), proxyres.affine)
                mask_resize = (
                        vol_resample_fast(proxyraw, mask_proxy, return_np=True) > 0.5
                )
                #
                wm_mask_proxy = nib.Nifti1Image(
                    wm_mask.astype("float"), proxyres.affine
                )
                wm_mask_resize = (
                        vol_resample_fast(proxyraw, wm_mask_proxy, return_np=True) > 0.5
                )

                mri_acq_orig_corr = copy.copy(mri_acq_orig.astype("float32"))
                mri_acq_orig_corr[mask_resize] = (
                        mri_acq_orig_corr[mask_resize] / bias_field_resize[mask_resize]
                )

                m = np.mean(mri_acq_orig_corr[wm_mask_resize])
                mri_acq_orig_corr = 110 * mri_acq_orig_corr / m
                # mask_dilated = binary_dilation(mask_resize, ball(3))
                # mri_acq_orig_corr[mask_dilated == 0] = 0  # Removing the skull-stripped step

                save_volume(
                    volume=np.clip(mri_acq_orig_corr, 0, 255).astype("uint8"),
                    aff=proxyraw.affine,
                    header=proxyraw.header,
                    path=output_filepath,
                )

                del (
                    bias_field,
                    bias_field_resize,
                    mri_acq_orig,
                    mri_acq_orig_corr,
                )  # , mask_dilated

            return {
                "exit_code": 0,
                "message": "success",
            }

    def process_subject(self, subject: str, force_flag: bool=False, **kwargs):
        """Run bias-field correction for all sessions of one subject.

        Parameters
        ----------
        subject : str
            Subject ID.
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

        exit_dict = {"images_processed": [], "images_failed": [], "exit_code": 0, "message": ""}
        for sess_id in tqdm.tqdm(sessions, leave=False):
            # input segs
            synthseg_entities = copy.copy(self.seg_entities)
            synthseg_entities["scope"] = ["nicgiprep-cross"]
            synthseg_entities["suffix"] = ["T1wsynthseg", "synthseg"]
            synthseg_entities["datatype"] = ["utils"]
            seg_files = self._get_data(
                **{"session": sess_id, "subject": subject, **synthseg_entities}, ignore_check=True
            )
            if seg_files is None:
                continue

            for seg_file in seg_files:
                ret_code = self.process_image(seg_file=seg_file, force_flag=force_flag)
                if ret_code['exit_code'] == 0:
                    exit_dict['images_processed'] += [seg_file.path]
                else:
                    exit_dict['images_failed'] += [seg_file.path]
                    exit_dict['exit_code'] = 1

        exit_dict['total_images'] = len(exit_dict['images_processed']) + len(exit_dict['images_failed'])
        exit_dict['message'] = (str(len(exit_dict['images_processed'])) + "/" +
                                str(exit_dict['total_images']) +
                                " of images were successfully processed.")
        print(' o ' + exit_dict['message'])
        print("\n")
        return exit_dict


class MNIRegistrationProcessor(CrossSectionalProcessor):
    """MNI regstration for cross-sectional data.

    For each session, register the bias-field-corrected brain image using
    its resampled SynthSeg parcellation and the MNI template. Saves both
    the registered brain and the affine array.
    """

    def _name(self):
        """Return the display name of this pipeline."""
        return "MNIRegistration"

    def process_subject(self, subject: str, force_flag: bool=False, **kwargs):
        """Run mni-registration for all sessions of one subject.

        Parameters
        ----------
        subject : str
            Subject ID.
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
        """
        print("\nSubject: " + subject)

        sessions = self._get_sessions(subject=subject)
        exit_dict = {"images_processed": [], "images_failed": [], "exit_code": 0, "message": ""}
        for sess_id in tqdm.tqdm(sessions, leave=False):

            # preproc_dirname = join(
            #     DIR_PIPELINES["nicgiprep-cross"],
            #     "sub-" + subject,
            #     "ses-" + sess_id,
            #     "anat",
            # )
            # if not exists(preproc_dirname):
            #     os.makedirs(preproc_dirname)

            # input segs
            synthseg_entities = copy.copy(self.seg_entities)
            synthseg_entities["scope"] = ["nicgiprep-cross"]
            synthseg_entities["datatype"] = ["utils"]
            synthseg_entities["suffix"] = ["T1wsynthseg", "synthseg"]
            seg_files = self._get_data(
                **{"session": sess_id, "subject": subject, **synthseg_entities}, ignore_check=True
            )
            if seg_files is None:
                continue

            for seg_file in seg_files:
                # get entities
                seg_entities = self._get_entities(seg_file)
                seg_entities["extension"] = ".nii.gz"

                raw_entities = copy.deepcopy(seg_entities)
                raw_entities["suffix"] = "T1w"
                raw_entities["datatype"] = "anat"
                raw_entities["acquisition"] = [None, 'orig']
                raw_entities["scope"] = "nicgiprep-cross"
                raw_entities["space"] = [None, 'subject']
                raw_entities.pop("datatype", None)
                if 'run' not in raw_entities: raw_entities['run'] = None

                # raw image
                raw_file = self._get_data(**raw_entities, curr_len=1)
                if raw_file is None:
                    exit_dict['images_failed'] += [seg_file.path]
                    continue

                # build output paths
                output_entities = copy.deepcopy(raw_entities)
                output_entities['desc'] = 'affine'
                output_entities['space'] = 'MNI'
                output_entities['acquisition'] = None
                output_filename = self.build_path(output_entities)
                output_filepath = join(DIR_PIPELINES["nicgiprep-cross"], output_filename)  ## Saving the final result in /anat

                output_entities['suffix'] = 'aff'
                output_entities['desc'] = None
                output_entities['extension'] = '.npy'
                output_aff_filename = self.build_path(output_entities)
                output_aff_filepath = join(DIR_PIPELINES["nicgiprep-cross"], output_aff_filename)  ## Saving the final result in /anat

                if exists(output_filepath) and not force_flag:
                    exit_dict['images_processed'] += [seg_file.path]
                    continue

                # ------------------------ #
                #   Registration to MNI    #
                # ------------------------ #
                im_MNI_proxy, aff_MNI_arr = register_to_MNI(
                    raw_file.path,
                    seg_file.path,
                    MNI_TEMPLATE_SEG,
                    MNI_TEMPLATE,
                    labels_registration,
                )

                nib.save(im_MNI_proxy, output_filepath)
                np.save(output_aff_filepath, aff_MNI_arr)
                exit_dict['images_processed'] += [seg_file.path]

        exit_dict['total_images'] = len(exit_dict['images_processed']) + len(exit_dict['images_failed'])
        exit_dict['message'] = (str(len(exit_dict['images_processed'])) + "/" +
                                str(exit_dict['total_images']) +
                                " of images were successfully processed.")
        print(' o ' + exit_dict['message'])
        print("\n")
        return exit_dict
