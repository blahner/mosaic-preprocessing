from dotenv import load_dotenv
load_dotenv()
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import argparse
from glmsingle.glmsingle import GLM_single
import nibabel as nib
import glob as glob
import pickle

#local
from src.utils.helpers import interpolate_ts

#compute glm using glmsingle
# https://github.com/cvnlab/GLMsingle

def ceildiv(a, b):
    return -(a // -b)

def main(args):
    subject = f"sub-{int(args.subject):02}"
    session = f"ses-nsdsynthetic"
    if args.verbose:
        print(f"Running GLMsingle on NSD main task for subject {subject} session {session}")

    TR = 1.6 #acquisition TR for NSD
    TR_resamp = 1 # resample time series to be locked to stimulus onset
    stimDur = 2 #in seconds
    dummy_offset = 0 #offset of start. in seconds

    numruns = len(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-*_run-*_desc-confounds_timeseries.tsv")))  #nothing special about the confounds file choice. should be 8 runs
    assert numruns == 8, f"Found only {numruns} runs for {subject} {session}. Should be 8."
    if args.verbose:
        print(f"Found {numruns} runs for subject {subject} session {session}")

    data = []
    design = []
    cols = ['trial_type','onset']
    events_run = []     
    ##Load eventts and data for each run
    ses_conds = [] #keep track of the test conditions shown in this session over all runs
    for count, run in enumerate(range(1,numruns+1)):
        if run%2 == 0: #if run is even
            task = 'memory' #or one-back
        else:
            task = 'fixation'
        task_run = ceildiv(run, 2)
        if args.verbose:
            print(f"{task} run number {task_run} (run {run}/{numruns})")
        #load data
        fmri_img = nib.load(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-{task_run:02}_space-fsLR_den-91k_bold.dtseries.nii")).get_fdata()
        #interpolate time series
        fmri_img_interp = interpolate_ts(fmri_img.T, TR, TR_resamp)
        numscans_interp = fmri_img_interp.shape[1]
        data.append(fmri_img_interp)

        #load events
        events_tmp = {col: [] for col in cols}  
        tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{task_run:02}_events.tsv"))
        prev_stim = np.nan
        for idx, stimIDX in enumerate(tmp.loc[:,'stim_idx']):
            if stimIDX == prev_stim:
                continue #skip one-back stimuli
            if stimIDX not in ses_conds:
                ses_conds.append(stimIDX)
            onset = tmp.loc[idx,'onset']
            events_tmp['trial_type'].append(stimIDX)
            events_tmp['onset'].append(onset + dummy_offset)
            prev_stim = stimIDX #update the previous stimuli for next row
        events_run.append(events_tmp)

    #create the design matrix for all runs in the session
    for count, run in enumerate(range(1,numruns+1)):
        run_design = np.zeros((numscans_interp, len(ses_conds)))
        events = events_run[count]
        for c, cond in enumerate(ses_conds):
            if cond not in events['trial_type']:
                continue #this stimulus was not presented in this run, so skip to next stimulus
            condidx = np.argwhere(np.array(events['trial_type'])==cond)[:,0]
            onsets_t = np.array(events['onset'])[condidx]
            onsets_tr = np.round(onsets_t / TR_resamp).astype(int)
            run_design[onsets_tr, c] = 1 
        design.append(run_design)

    #define opt for glmsingle params
    opt = dict()
    # set important fields for completeness (but these would be enabled by default)
    opt['wantlibrary'] = 1
    opt['wantglmdenoise'] = 1
    opt['wantfracridge'] = 1

    # for the purpose of this example we will keep the relevant outputs in memory
    # and also save them to the disk
    opt['wantfileoutputs'] = [0,0,0,1]
    opt['wantmemoryoutputs'] = [0,0,0,1]
    numvertices = data[0].shape[0]  # get shape of data for convenience
    opt['chunklen'] = int(numvertices) 

    outputdir_glmsingle = os.path.join(args.dataset_root, "derivatives", "GLM", subject, session)
    if not os.path.exists(outputdir_glmsingle):
        os.makedirs(outputdir_glmsingle)

    if args.verbose:
        print(f"running GLMsingle...")
    start_time = time.time()

    glmsingle_obj = GLM_single(opt)
    # run GLMsingle
    glmsingle_obj.fit(
        design,
        data,
        stimDur,
        TR_resamp,
        outputdir=outputdir_glmsingle)

    #save design matrix
    if args.verbose:
        print("saving design matrix")
    with open(os.path.join(outputdir_glmsingle, f"{subject}_{session}_conditionOrderDM.pkl"), 'wb') as f:
        pickle.dump((events_run, ses_conds), f)

    elapsed_time = time.time() - start_time
    if args.verbose:
        print(
            '\telapsed time: ',
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
        )

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"NaturalScenesDataset") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-8 that you wish to process")
    parser.add_argument("-d", "--dataset_root",  default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
