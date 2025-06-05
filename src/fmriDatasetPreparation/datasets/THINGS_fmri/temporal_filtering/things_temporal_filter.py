from dotenv import load_dotenv
load_dotenv()
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import nibabel as nib
import subprocess
import argparse
from nilearn.image import clean_img
import glob as glob

def main(args):
    if args.task not in ['things', 'rest', '6cat', 'pRF']:
        raise ValueError(f"task not recognized. You entered {args.task} but it must be one of 'things', 'rest', '6cat', 'pRF'")

    #define subject and session
    subject = f"sub-{int(args.subject):02}"

    #get the names of all sessions for this subject
    sessions_all = glob.glob(os.path.join(args.dataset_root, "derivatives","fmriprep",subject, '*'))
    #search over all sessions to find the ones that include runs of the specified task
    sessions_contains_task = [] #compile all the session numbers that have at least one run from this task
    for session_path in sessions_all:
        session = session_path.split('/')[-1]
        numruns = len(glob.glob(os.path.join(session_path, "func", f"{subject}_{session}_task-{args.task}_*_desc-confounds_timeseries.tsv")))
        if numruns > 0:
            sessions_contains_task.append(session)
    assert(len(sessions_contains_task) > 0)
    print(f"Found {args.task} runs in {len(sessions_contains_task)} sessions: {sessions_contains_task}")

    COI = ['trans_x','trans_x_derivative1', 'trans_x_power2', 'trans_x_derivative1_power2',
        'trans_y', 'trans_y_derivative1', 'trans_y_power2', 'trans_y_derivative1_power2',
        'trans_z', 'trans_z_derivative1', 'trans_z_power2', 'trans_z_derivative1_power2',
        'rot_x', 'rot_x_derivative1', 'rot_x_power2', 'rot_x_derivative1_power2',
        'rot_y', 'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2',
        'rot_z', 'rot_z_derivative1', 'rot_z_power2', 'rot_z_derivative1_power2'] #confounds of interest to include. HCP resting state: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3720828/ and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3811142/
     
    start_from_TR = 0 #in case you have dummy TRs in the beginning
    TR_acq=1.5
    
    for session in sessions_contains_task:
        fmri_output_path = os.path.join(args.dataset_root, "derivatives", "temporal_filtering", subject, session)
        if not os.path.exists(fmri_output_path):
            os.makedirs(fmri_output_path)

        numruns = len(glob.glob(os.path.join(args.dataset_root, "derivatives","fmriprep",subject, session, "func", f"{subject}_{session}_task-{args.task}_*_desc-confounds_timeseries.tsv")))
        assert(numruns > 0)

        for r in range(1, numruns+1): #each run is temporally filtered independently

            print(f"starting {subject} {session} task {args.task} run {r}")
            #path to timeseries file from fmriprep output
            func_root = os.path.join(args.dataset_root, 'derivatives','fmriprep', subject, session, 'func')
            if args.task == 'rest':
                body = f"task-{args.task}_acq-reversePE"
            else:
                body = f"task-{args.task}_run-{r:02}"

            cifti_ts_path = os.path.join(func_root, f'{subject}_{session}_{body}_space-fsLR_den-91k_bold.dtseries.nii')

            cifti_ts_clean_path = os.path.join(fmri_output_path, f'{subject}_{session}_{body}_space-fsLR_den-91k_bold_clean.dtseries.nii')
            if os.path.isfile(cifti_ts_clean_path):
                print("Run already cleaned and saved. Skipping to next one...")
                continue

            #load nuissance regressors
            confounds = pd.read_table(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_{body}_desc-confounds_timeseries.tsv"))
            confounds_select = confounds.loc[:,COI] #(numscans_interp, numregressors)
            motion_regressors = np.array(confounds_select)
            if (~np.isfinite(motion_regressors)).sum() > 0:
                motion_regressors[~np.isfinite(motion_regressors)] = 0 #fill nans or infs with 0

            tmpdir = os.path.join(fmri_output_path, "tmp")
            input_nifti = os.path.join(tmpdir,'func_fnifti.nii.gz') #create a fake nifti
            cmd = f"wb_command -cifti-convert -to-nifti -smaller-dims \
                {cifti_ts_path} \
                {input_nifti}"

            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)
            subprocess.run(cmd, shell=True, check=True)

            #index non-dummy TRs
            nib_image = nib.load(input_nifti)
            trimmed_nifti = nib_image.slicer[:,:,:, start_from_TR:]

            #clean the data using nilearn
            clean_output = clean_img(trimmed_nifti, detrent=True, standardize=True, confounds=motion_regressors, high_pass=0.008, t_r=TR_acq, ensure_finite=False) #no low pass, like in HCP. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3811142/
            clean_output_nifti = os.path.join(tmpdir, 'clean_fnifti.nii.gz')
            clean_output.to_filename(clean_output_nifti)

            #Go back from nifti to cifti
            """
            documentation: https://www.humanconnectome.org/software/workbench-command/-cifti-convert
            nifti-in: our input nifti file that has been cleaned and saved to a tmp file called 'clean_fnifti.nii.gz'
            cifti-template: our original cifti file for the data that we now use as a template
            cifti-out: path and filename for the new saved nifti file

            """

            cmd = f"wb_command -cifti-convert -from-nifti \
                {clean_output_nifti} \
                {cifti_ts_path} \
                {cifti_ts_clean_path} \
                '-reset-timepoints' \
                {TR_acq} \
                {start_from_TR}"

            subprocess.run(cmd, shell=True, check=True)
            print(f"success: conversion from clean nifti to clean cifti for run {r}")

            #delete tmp directory
            subprocess.run(f"rm -r {tmpdir}", shell=True, check=True)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"THINGS_fmri") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-3 that you wish to process")
    parser.add_argument("-t", "--task", type=str, required=True, help="What task ('things', 'rest', '6cat', 'pRF') to analyze)")
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
    args = parser.parse_args()

    main(args)