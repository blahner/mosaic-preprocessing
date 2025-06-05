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
from nilearn.glm.first_level import compute_regressor

#local
from src.utils.helpers import interpolate_ts

#compute glm using glmsingle
# https://github.com/cvnlab/GLMsingle

def main(args):
    subject = f"sub-{int(args.subject):02}"
    if args.verbose:
        print(f"starting subject: {subject}")
    
    session_group = f"sessiongroup-01" #for all subjects there will only be one session group.

    #search over all sessions to find the ones that include runs of the specified task
    session_path = os.path.join(args.dataset_root, "derivatives", "fmriprep", subject)
    sessions_tmp = sorted(glob.glob(os.path.join(session_path, f"*imagenet*"))) + sorted(glob.glob(os.path.join(session_path, f"*coco*")))  #compile all the session numbers
    assert(len(sessions_tmp) > 0)
    sessions = []
    for s in sessions_tmp:
        sname = s.split("/")[-1]
        if "imagenet05" not in sname:
            sessions.append(sname)
    assert(len(sessions) > 0)
    if args.verbose:
        print(f"Found {len(sessions)} sessions")
        print(f"{sessions}")

    """
    Same dummy offset for imagenet and coco according glm code (https://github.com/BNUCNL/NOD-fmri/blob/main/validation/GLM.py), 
    but main manuscript text says coco has 4 trials (=12s) of beginning and ending rest. Also main manuscript says 
    each coco session has 10 runs, but subjects 2-9 have 11 runs
    """
    TR = 2 #acquisition TR in seconds for NOD

    #The actual stimDur for Coco sessions is 0.5 and imagenet is 1s, but to combine coco and imagenet with glmsingle we need to make it the same. Keep this standard for subs 10-30 even though no coco is available.
    TR_resamp = 0.5 # resample time series to be locked to stimulus onset
    stimDur = 0.5 #in seconds

    data = []
    design = []
    allsession_conds = [] 
    cols = ['trial_type','onset'] #columns for events file
    events_run = []    
    session_indicator = []

    #if args.subject >= 10:
    #subjects 10-30 dont have a coco session to take advantage of repeats for cross-validated nuissance regressors, so use motion regressors
    COI = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'] #confounds of interest to include in design matrix
    nuissance_regressors = [] 

    for session_count, session_path in enumerate(sessions):
        session = session_path.split('/')[-1]
        if 'coco' in session:
            task='coco'
            if subject == 'sub-01':
                events_stim_field = 'stim_file'
            else:
                events_stim_field = 'condition'
            dummy_offset = 16 #offset of start in seconds. Coco session onsets need to be offset by 16 sec
        elif 'imagenet' in session:
            task='imagenet'
            events_stim_field = 'stim_file'
            dummy_offset = 0 #16 #offset of start in seconds. Imagenet session onsets in the event files are already offset by 16 sec
        else:
            raise ValueError("Invalid task name. Must be either coco or imagenet session.")
        
        if args.verbose:
            print(f"Running GLMsingle on NOD {task} task for subject {subject} session {session}")
        numruns = len(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-*_desc-confounds_timeseries.tsv")))  #nothing special about the confounds file choice
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
            if task == 'coco':
                assert(fmri_img.shape == (241, 91282))
            #interpolate time series
            fmri_img_interp = interpolate_ts(fmri_img.T, TR, TR_resamp)
            numscans_interp = fmri_img_interp.shape[1]

            data.append(fmri_img_interp)

            #load events
            events_tmp = {col: [] for col in cols}
            button_press_right = []
            button_press_left = []
            tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
            for idx, stim_id in enumerate(tmp.loc[:,events_stim_field]):
                stimIDX = stim_id
                if str(stimIDX) == 'nan': #for coco subject 1, blank trials are not input into the events file. for subjects 2-9, they are listed as n/a conditions with their own onsets
                    continue
                if stimIDX not in allsession_conds:
                    allsession_conds.append(stimIDX)
                onset = tmp.loc[idx,'onset']
                events_tmp['trial_type'].append(stimIDX)
                events_tmp['onset'].append(onset + dummy_offset)
                if task == 'imagenet': #only imagenet has button presses during stimulus presentation and coco event files dont have a button press field
                    response_time = tmp.loc[idx,'response_time'] #time after stimulus onset that the button was pressed. 0.0 is if no response was detected
                    response = tmp.loc[idx, 'response'] #1 for animate (right thumb) or -1 for inanimate (left thumb) 
                    if response != 0: #a 'response' of 0.0 and 'response_time' of 0.0 means the subject did not press the button.
                        response_time_run = onset + dummy_offset + response_time #response time relative to start of run
                        if response == 1:
                            button_press_right.append(response_time_run) 
                        elif response == -1:
                            button_press_left.append(response_time_run)
                        else:
                            raise ValueError(f"Button press value not recognized. Should be -1 or 1 but saw {response}.")
            events_run.append(events_tmp)

            nuissance_list = [] #collects motion and button press nuissance regressors for this run, if applicable/available
            #subjects 1-9 should imagenet runs should have button press regressors, subs 10-30 imagenet should have motion and button press regressors
            if args.subject >= 10:
                #subjects 10-30 dont have a coco session to take advantage of repeats for cross-validated nuissance regressors, so use motion regressors
                #load nuissance regressors
                confounds = pd.read_table(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_desc-confounds_timeseries.tsv"))
                confounds_select = confounds.loc[:,COI].T 
                confounds_interp = interpolate_ts(confounds_select, TR, TR_resamp)
                nuissance_list.append(confounds_interp.T)

            if task == 'imagenet':
                #convolve the button press with HRF to compute a regressor
                #expects exp input to be of shape (3,n) specifying onsets, durations, amplitudes by time
                bp_condition_right = [] # np.zeros(3,button_press.sum())
                bp_condition_left = []
                bp_duration = 0 #impulse duration
                bp_amplitute = 1 #amplitude of 1
                for bp_onset_r in button_press_right:
                    bp_condition_right.append([bp_onset_r, bp_duration, bp_amplitute])
                for bp_onset_l in button_press_left:
                    bp_condition_left.append([bp_onset_l, bp_duration, bp_amplitute])
                if len(bp_condition_right) > 0: #sometimes the subject forgot to press the button at all during the run.
                    bp_condition_right = np.vstack(bp_condition_right).T
                    frame_times = np.array(np.arange(0,fmri_img.shape[0]*TR, TR_resamp)) #frame times in seconds, shape (nscans)
                    button_press_regressor_right = compute_regressor(bp_condition_right, hrf_model='spm', frame_times=frame_times)[0] #numscans_interp x numregressors
                    nuissance_list.append(button_press_regressor_right)
                if len(bp_condition_left) > 0:
                    bp_condition_left = np.vstack(bp_condition_left).T
                    frame_times = np.array(np.arange(0,fmri_img.shape[0]*TR, TR_resamp)) #frame times in seconds, shape (nscans)
                    button_press_regressor_left = compute_regressor(bp_condition_left, hrf_model='spm', frame_times=frame_times)[0] #numscans_interp x numregressors
                    nuissance_list.append(button_press_regressor_left)

            if nuissance_list:
                if len(nuissance_list) == 1:
                    #no need to stack
                    nuissance_regressors.append(np.array(nuissance_list[0]))
                else:
                    nuissance_regressors.append(np.hstack(nuissance_list))
            else:
                nuissance_regressors.append(np.array([]))

    #create the design matrix for all runs in the combined sessions
    for count, _ in enumerate(range(1,len(data)+1)):
        numscans_interp = data[count].shape[1]
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
    if args.subject >= 10: #imagenet sessions do not have any repeats (within or between sessions), so we can only do type b glmsingle
        opt = dict()
        # set important fields for completeness (but these would be enabled by default)
        opt['wantlibrary'] = 1
        opt['wantglmdenoise'] = 0
        opt['wantfracridge'] = 0

        # for the purpose of this example we will keep the relevant outputs in memory
        # and also save them to the disk
        opt['wantfileoutputs'] = [0,1,0,0]
        opt['wantmemoryoutputs'] = [0,1,0,0]
        opt['extra_regressors'] = nuissance_regressors #this includes the button presses
    elif args.subject < 10: #the coco session has repeated stimuli within a session, so we can do type d glmsingle
        opt = dict()
        # set important fields for completeness (but these would be enabled by default)
        opt['wantlibrary'] = 1
        opt['wantglmdenoise'] = 1
        opt['wantfracridge'] = 1

        # for the purpose of this example we will keep the relevant outputs in memory
        # and also save them to the disk
        opt['wantfileoutputs'] = [0,0,0,1]
        opt['wantmemoryoutputs'] = [0,0,0,1]
        opt['extra_regressors'] = nuissance_regressors

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
    with open(os.path.join(outputdir_glmsingle, f"{subject}_{session_group}_conditionOrderDM.pkl"), 'wb') as f:
        pickle.dump((events_run, allsession_conds), f)

    elapsed_time = time.time() - start_time
    if args.verbose:
        print(
            '\telapsed time: ',
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
        )

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"NaturalObjectDataset") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-10 that you wish to process")
    parser.add_argument("-d", "--dataset_root",  default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
