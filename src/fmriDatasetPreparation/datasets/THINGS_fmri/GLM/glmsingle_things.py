from dotenv import load_dotenv
load_dotenv()
import numpy as np
import scipy
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

def main(args):
    subject = f"sub-{int(args.subject):02}"
    if args.verbose:
        print(f"starting subject: {subject}")
    session_groups = {"sub-01": [[1,2,3,4,5,6],[7,8,9,10,11,12]],
                      "sub-02": [[1,2,3,4,5,6],[7,8,9,10,11,12]],
                      "sub-03": [[1,2,3,4,5,6],[7,8,9,10,11,12]],
                      }
    sessions = session_groups[subject][args.session_group-1]
    session_group = f"sessiongroup-{args.session_group:02}"
    task = 'things'
    print("#"*10)
    print(f"Starting GLMsingle for THINGS_fmri {subject} task {task} combining sessions {sessions} ({session_group}).")

    #things has a 0.5s stim duration followed by a 4s fixation (4.5 s total.)
    #100 test images were presented 12x but only once within a session, so here we use typeb glmsingle.
    #the vigilance task was to button press on a GAN generated catch image, not included in GLM.
    #720 trials per session (72 trials per run, 10 runs)
    TR = 1.5 #acquisition TR in seconds for THINGS fmri

    TR_resamp = 0.5 # resample time series to be locked to stimulus onset
    stimDur = 0.5 #in seconds

    dummy_offset = 0 #offset of start in seconds. rest at beggining of session is included in the events file onsets.
    data = []
    design = []
    allsession_conds = [] 
    cols = ['trial_type','onset'] #columns for events file
    events_run = []    
    session_indicator = []
    
    for session_count, session_num in enumerate(sessions): #concatenate sessions
        session = f"ses-{task}{int(session_num):02}" #only valid for things task right now. localizer task doesnt have 0 padding
        numruns = len(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-*_desc-confounds_timeseries.tsv")))  #nothing special about the confounds file choice
        assert(numruns > 0)
        if args.verbose:
            print(f"Found {numruns} runs for subject {subject} session {session}")
   
        ##Load eventts and data for each run
        for count, run in enumerate(range(1,numruns+1)):
            if args.verbose:
                print("run:",run)
            session_indicator.append(session_count+1) #indicate to glmsingle that all these runs belong to the same session for internal normalization
            #load data
            fmri_img = nib.load(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_space-fsLR_den-91k_bold.dtseries.nii")).get_fdata()

            #interpolate time series
            fmri_img_interp = interpolate_ts(fmri_img.T, TR, TR_resamp)
            numscans_interp = fmri_img_interp.shape[1]
            data.append(fmri_img_interp)

            #load events
            events_tmp = {col: [] for col in cols}  
            tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
            for idx, img_filename in enumerate(tmp.loc[:,'file_path']):
                if tmp.loc[idx,'trial_type'] == 'catch':
                    continue
                if img_filename not in allsession_conds:
                    allsession_conds.append(img_filename)
                onset = tmp.loc[idx,'onset']
                events_tmp['trial_type'].append(img_filename)
                events_tmp['onset'].append(onset + dummy_offset)
            events_run.append(events_tmp)

    #create the design matrix for all runs in the session
    for count, run in enumerate(range(1,len(session_indicator)+1)):
        run_design = np.zeros((numscans_interp, len(allsession_conds)))
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

    outputdir_glmsingle = os.path.join(args.dataset_root, "derivatives", "GLM", subject, session_group, task)
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
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"THINGS_fmri") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-3 that you wish to process")
    parser.add_argument("-i", "--session_group", type=int, required=True, help="The sessions you want to perform glmsingle on (1-indexed).")
    parser.add_argument("-d", "--dataset_root",  default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
