from dotenv import load_dotenv
load_dotenv()
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pickle
import os
import argparse
from glmsingle.glmsingle import GLM_single
import nibabel as nib
import glob as glob

#local
from src.utils.helpers import interpolate_ts

#compute glm using glmsingle
# https://github.com/cvnlab/GLMsingle

"""
As done in GLMsingle (Prince et al., 2022), we need to combine sessions to capture repeated stimuli
to take advantage of GLMsingle's cross validation dependent features (types c and d). In subjects
1, 2, and 3, 5 sessions are grouped together (total of 15 sessions). In subject 4, the first 5 
sessions are grouped and the last 4 sessions are grouped (total of 9 sessions). GLMsingle's 
sessionindicator is used to internally normalize sessions.
"""

def main(args):
    subject = f"sub-CSI{int(args.subject)}"
    session_groups = {"sub-CSI1": [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]],
                      "sub-CSI2": [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]],
                      "sub-CSI3": [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]],
                      "sub-CSI4": [[1,2,3,4,5],[6,7,8,9]]
                      }
    sessions = session_groups[subject][args.session_group-1]
    session_group = f"sessiongroup-{args.session_group:02}"
    task="5000scenes"

    print("#"*10)
    print(f"Starting GLMsingle for BOLD5000 {subject} task {task} combining sessions {sessions} ({session_group}).")

    TR = 2 #acquisition TR
    TR_resamp = 1 # resample time series to be locked to stimulus onset
    stimDur = 1 #in seconds
    dummy_offset = 0 #offset of start. in seconds. 6s rest at beginning of scan is included in the events file
    
    data = []
    design = []
    allsession_conds = [] 
    cols = ['trial_type','onset'] #columns for events file
    events_run = []    
    session_indicator = []
    for session_count, session_num in enumerate(sessions): #concatenate sessions
        session = f"ses-{int(session_num):02}"

        sub_func_root = os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func")
        numruns = len(glob.glob(os.path.join(sub_func_root, f"*task-{task}_*confounds_timeseries.tsv")))
        assert(numruns > 0)
        if args.verbose:
            print(f"Found {numruns} runs for subject {subject} session {session} task {task}. Running GLMsingle.")
 
        ##Load events and data for each run
        for count, run in enumerate(range(1,numruns+1)):
            if args.verbose:
                print("run:",run)
            session_indicator.append(session_count+1) #indicate to glmsingle that all these runs belong to the same session for internal normalization
            #load data
            cifti_ts = nib.load(os.path.join(sub_func_root, f"{subject}_{session}_task-{task}_run-{run:02}_space-fsLR_den-91k_bold.dtseries.nii"))
            cifti_data = cifti_ts.get_fdata()

            #interpolate time series
            fmri_interp = interpolate_ts(cifti_data.T, TR, TR_resamp)
            numscans_interp = fmri_interp.shape[1]
            data.append(fmri_interp)

            #load events
            events_tmp = {col: [] for col in cols}  
            tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
            for idx, fname in enumerate(tmp.loc[:,'stim_file']):
                if fname not in allsession_conds:
                    allsession_conds.append(fname)
                onset = tmp.loc[idx,'onset']
                events_tmp['trial_type'].append(fname)
                events_tmp['onset'].append(onset + dummy_offset)
            events_run.append(events_tmp)

    #create the design matrix for all runs in all sessions
    for count, run in enumerate(range(1,len(session_indicator)+1)):
        run_design = np.zeros((numscans_interp, len(allsession_conds)))
        events = events_run[count]
        for c, cond in enumerate(allsession_conds):
            if cond not in events['trial_type']:
                continue
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

    outputdir_glmsingle = os.path.join(args.dataset_root, "derivatives", "GLM", subject, session_group, task)
    if not os.path.exists(outputdir_glmsingle):
        os.makedirs(outputdir_glmsingle)

    start_time = time.time()
    if args.verbose:
        print(f"running GLMsingle...")
    #sometimes the default 50,000 chunk length doesn't chunk into equal lengths, throwing an error when converting to array
    numvertices = data[0].shape[0]  # get shape of data for convenience
    opt['chunklen'] = int(numvertices) 
    opt['sessionindicator'] = np.array(session_indicator)

    glmsingle_obj = GLM_single(opt)
    # run GLMsingle
    glmsingle_obj.fit(
        design,
        data,
        stimDur,
        TR_resamp,
        outputdir=outputdir_glmsingle)
    elapsed_time = time.time() - start_time

    if args.verbose:
        print(
            '\telapsed time: ',
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
        )
    #save design matrix
    if args.verbose:
        print("saving design matrix")
    with open(os.path.join(outputdir_glmsingle, f"{subject}_{session_group}_task-{task}_conditionOrderDM.pkl"), 'wb') as f:
        pickle.dump((events_run, allsession_conds), f)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"BOLD5000") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-4 that you wish to process")
    parser.add_argument("-i", "--session_group", type=int, required=True, help="The sessions you want to perform glmsingle on (1-indexed). For subjects 1-3, the session_group = 1 is sessions 1-5, 2 is 6-10, and 3 is 11-15. For subject 4, group 1 is 1-5, group 2 is 6-9.")
    parser.add_argument("-d", "--dataset_root",  default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
    args = parser.parse_args()

    main(args)