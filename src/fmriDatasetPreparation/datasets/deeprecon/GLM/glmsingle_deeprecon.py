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

#compute glm using glmsingle
# https://github.com/cvnlab/GLMsingle

def main(args):
    subject = f"sub-{int(args.subject):02}"
    if args.session_group in ['perceptionArtificialImage','perceptionLetterImage','perceptionNaturalImageTest','perceptionNaturalImageTraining']:
        task = 'perception'
    elif args.session_group in ['imagery']:
        task = 'imagery'
        raise ValueError("imagery task not yet fully supported")
    #search over all sessions to find the ones that include runs of the specified task
    sessions = sorted(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, f"*{args.session_group}*"))) #compile all the session numbers that have at least one run from this task
    assert(len(sessions) > 0)
    session_name = []
    for s in sessions:
        session_name.append(s.split("/")[-1])
    assert(len(session_name) > 0)
    if args.verbose:
        print(f"Found {len(sessions)} sessions for session group {args.session_group}")
        print(f"sessions: {session_name}")

    session_group = f"sessiongroup-{args.session_group}" #combine all sessions within the task

    TR = 2 #acquisition TR in seconds for deeprecon.
    TR_resamp = TR # resample time series to be locked to stimulus onset. If equal to the TR, no resampling takes place

    if args.session_group in ['perceptionArtificialImage','perceptionNaturalImageTest','perceptionNaturalImageTraining', 'imagery']:
        stimDur = 8 #in seconds
    elif args.session_group in ['perceptionLetterImage']: #letter image task also has 12s of rest after the block presentation
        stimDur = 12 #in seconds

    dummy_offset = 0 # offset of start in seconds. 33s of rest at beginning of sessions but this is accounted for in the events file. 6s rest at end of sessions. 9s of dummy were excluded already. 8s of dummy were excluded for retinotopy experiment
    data = []
    design = []
    allsession_conds = [] 
    cols = ['trial_type','onset']
    events_run = [] 
    session_indicator = []

    for session_count, session in enumerate(session_name):
        numruns = len(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-*{task}*_run-*_desc-confounds_timeseries.tsv")))  #nothing special about the confounds file choice
        assert(numruns > 0)
        if args.verbose:
            print(f"Found {numruns} runs for subject {subject} session {session}")
   
        ##Load eventts and data for each run
        for count, run in enumerate(range(1,numruns+1)):
            if args.verbose:
                print("run:",run)
            session_indicator.append(session_count+1)
            #load data
            fmri_img = nib.load(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_space-fsLR_den-91k_bold.dtseries.nii")).get_fdata()
            numscans = fmri_img.shape[0] #fmri_img is shape numscans x numvertices
            data.append(fmri_img.T)

            #load events
            events_tmp = {col: [] for col in cols}  
            tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
            stim_id_prev = 0
            for idx, stim_id in enumerate(tmp.loc[:,'stimulus_id']):
                if np.isnan(stim_id):
                    continue #dont include resting state scans at beginning or end or rests between blocks
                if (stim_id == stim_id_prev):
                    stim_id_prev = stim_id
                    continue #dont include the one-back repeated stimuli
                stimIDX = stim_id
                if stimIDX not in allsession_conds:
                    allsession_conds.append(stimIDX)
                onset = tmp.loc[idx,'onset']
                events_tmp['trial_type'].append(stimIDX)
                events_tmp['onset'].append(onset + dummy_offset) # type: ignore
                stim_id_prev = stim_id
            events_run.append(events_tmp)

    #create the design matrix for all runs in the session
    for count, _ in enumerate(range(1,len(data)+1)):
        numscans = data[count].shape[1]
        run_design = np.zeros((numscans, len(allsession_conds)))
        events = events_run[count]
        for c, cond in enumerate(allsession_conds):
            if cond not in events['trial_type']:
                continue #this stimulus was not presented in this run, so skip to next stimulus
            condidx = np.argwhere(np.array(events['trial_type'])==cond)[:,0]
            onsets_t = np.array(events['onset'])[condidx]
            onsets_tr = np.round(onsets_t / TR_resamp).astype(int)
            run_design[onsets_tr, c] = 1 
        design.append(run_design)

    numvertices = data[0].shape[0]  # get shape of data for convenience
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

    opt['chunklen'] = int(numvertices) 
    opt['sessionindicator'] = np.array(session_indicator)

    outputdir_glmsingle = os.path.join(args.dataset_root, "derivatives", "GLM", subject, session_group)
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
    with open(os.path.join(outputdir_glmsingle, f"{subject}_{session_group}_task-{task}_conditionOrderDM.pkl"), 'wb') as f:
        pickle.dump((events_run, allsession_conds), f)

    elapsed_time = time.time() - start_time
    if args.verbose:
        print(
            '\telapsed time: ',
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
        )

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"deeprecon") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-3 that you wish to process")
    parser.add_argument("-i", "--session_group", type=str, required=True, help="The session group you are running the GLM for. must be one of ['perceptionArtificialImage','perceptionLetterImage','perceptionNaturalImageTest','perceptionNaturalImageTraining','imagery'].")
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()
    
    main(args)
