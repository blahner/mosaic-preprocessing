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
import nibabel as nib
from glmsingle.glmsingle import GLM_single
import pickle
from nilearn.glm.first_level import compute_regressor

#compute glm using glmsingle
# https://github.com/cvnlab/GLMsingle
#helpful nibabel + cifti tutorial: https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
#other cifti links:
# http://www.nitrc.org/projects/cifti/
# https://www.humanconnectome.org/software/workbench-command/-cifti-help

"""
    Parcels: each index refers to a named subset of the brainordinates (i.e.
        'V1', and the surface vertices in V1)
    Scalars: each index is simply given a name (i.e. 'Myelin')
    Series: each index is assigned a quantity in a linear series (i.e., a
        timeseries of 0 sec, 0.7 sec, 1.4 sec, ...)
    Labels: each index is assigned a name (i.e., 'Visual Areas'), but also a
        list of labels that maps integer data values to names and colors (i.e.
        {(5, 'V1', #ff0000), (7, 'V2', #00ff00), ...}

   The common types of cifti files and the mapping types they use are:
      dconn: ROW is dense, COLUMN is dense
      dscalar: ROW is scalars, COLUMN is dense
      dtseries: ROW is series, COLUMN is dense
      dlabel: ROW is labels, COLUMN is dense
      pconn: ROW is parcels, COLUMN is parcels
      pdconn: ROW is dense, COLUMN is parcels
      dpconn: ROW is parcels, COLUMN is dense
      pscalar: ROW is scalars, COLUMN is parcels
      ptseries: ROW is series, COLUMN is parcels

"""

"""
load data
cifti = nb.load(fname)
cifti_data = cifti.get_fdata(dtype=np.float32)
cifti_hdr = cifti.header
nifti_hdr = cifti.nifti_header

save the data
img = cifti2.Cifti2Image(cifti_data, header=cifti_hdr)
img.to_filename(’/path/test.dtseries.nii’)

"""

def main(args):
    subject = f"sub-{int(args.subject):02}"
    session = "ses-action01" #HAD only has one session called "action01"
    if args.verbose:
        print(f"Running GLMsingle on HAD dataset main task for subject {subject}")

    #getting some HAD parameters for the GLM here: https://github.com/BNUCNL/HAD-fmri/blob/main/validation/GLM.py
    #Theres some discrepancy with the manuscript saying "A clip was presented 2 seconds followed by a 2-second interval"
    # and "four blank trials added at the beginning and end of each run"
    #and "each run lasted 5 minutes and 12 seconds" (312 seconds). The nii files show 156 TRs (312 seconds), and the 
    #events tsv files onsets start at 0 and end with the last trial with an onset at 280sec. 4 blank trials is 16 seconds.
    #By my math a run should be: 16 seconds blank, 284 seconds stim, 16 seconds blank = 316sec (not 312!). Their code says stsim onsets
    #start after 12 seconds of blank, so then the math all adds up, but this means its a small typo in the manuscript.

    TR = 2 #acquisition TR
    stimDur = 2 #in seconds. 
    dummy_offset = 12 #offset of start. in seconds
    COI = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'] #confounds of interest to include in design matrix

    numruns = 12 #each subject has one session with 12 runs.

    data = []
    design = []
    nuissance_regressors=[]

    cols = ['trial_type','onset']
    events_run = []     
    ses_conds=[] #keep track of the conditions shown in this session over all 12 runs
    for count, run in enumerate(range(1,numruns+1)):
        if args.verbose:
            print("run:",run)
        #load data
        cifti = nib.load(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-action_run-{run:02}_space-fsLR_den-91k_bold.dtseries.nii"))
        cifti_data = cifti.get_fdata().T

        numscans = cifti_data.shape[1]
        data.append(cifti_data)

        #load events
        events_tmp = {col: [] for col in cols}  
        button_press_right = []
        button_press_left = []
        tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-action_run-{run:02d}_events.tsv"))
        for idx, tt in enumerate(tmp.loc[:,'stim_file']):
            stim_file_split = tt.split("/")
            fname = stim_file_split[1]
            fname_noext = fname.split(".mp4")[0] 
            assert(fname_noext not in ses_conds) #every video just shown once
            ses_conds.append(fname_noext)
            onset = tmp.loc[idx,'onset']
            events_tmp['trial_type'].append(fname_noext)
            events_tmp['onset'].append(onset + dummy_offset)
            response_time = tmp.loc[idx,'response_time']#time after stimulus onset that the button was pressed. 0.0 is if no response was detected
            response = tmp.loc[idx, 'response'] #1 for sport (right thumb) or -1 for non-sport (left thumb) 
            if not pd.isna(response_time) and not pd.isna(response):
                response_time_run = onset + dummy_offset + response_time #response time relative to start of run
                if response == 1:
                    button_press_right.append(response_time_run) 
                elif response == -1:
                    button_press_left.append(response_time_run)
        events_run.append(events_tmp)

        #load nuissance regressors
        nuissance_list = [] #add nuissance regressors
        confounds = pd.read_table(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-action_run-{run:02}_desc-confounds_timeseries.tsv"))
        confounds_select = confounds.loc[:,COI]
        nuissance_list.append(np.array(confounds_select))

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
            frame_times = np.array(np.arange(0,cifti.shape[0]*TR, TR)) #frame times in seconds, shape (nscans)
            button_press_regressor_right = compute_regressor(bp_condition_right, hrf_model='spm', frame_times=frame_times)[0] #numscans x numregressors
            nuissance_list.append(button_press_regressor_right)
        if len(bp_condition_left) > 0:
            bp_condition_left = np.vstack(bp_condition_left).T
            frame_times = np.array(np.arange(0,cifti.shape[0]*TR, TR)) #frame times in seconds, shape (nscans)
            button_press_regressor_left = compute_regressor(bp_condition_left, hrf_model='spm', frame_times=frame_times)[0] #numscans x numregressors
            nuissance_list.append(button_press_regressor_left)

        if nuissance_list:
            if len(nuissance_list) == 1:
                #no need to stack
                nuissance_regressors.append(np.array(nuissance_list[0]))
            else:
                nuissance_regressors.append(np.hstack(nuissance_list))
        else:
            nuissance_regressors.append(np.array([]))
            
    #create the design matrix for all runs in the session
    for count, run in enumerate(range(1,numruns+1)):
        run_design = np.zeros((numscans, len(ses_conds)))
        events = events_run[count]
        for c, cond in enumerate(ses_conds):
            if cond not in events['trial_type']:
                continue
            condidx = np.argwhere(np.array(events['trial_type'])==cond)[:,0]
            onsets_t = np.array(events['onset'])[condidx]
            onsets_tr = np.round(onsets_t / TR).astype(int)
            run_design[onsets_tr, c] = 1
        design.append(run_design)

    #define opt for glmsingle params
    opt = dict()
    # set important fields for completeness (but these would be enabled by default)
    opt['wantlibrary'] = 1
    opt['wantglmdenoise'] = 0
    opt['wantfracridge'] = 0

    # for the purpose of this example we will keep the relevant outputs in memory
    # and also save them to the disk
    opt['wantfileoutputs'] = [0,1,0,0]
    opt['wantmemoryoutputs'] = [0,1,0,0]
    opt['extra_regressors'] = nuissance_regressors

    outputdir_glmsingle = os.path.join(args.dataset_root, "derivatives", "GLM", subject, session)
    if not os.path.exists(outputdir_glmsingle):
        os.makedirs(outputdir_glmsingle)

    start_time = time.time()

    if args.verbose:
        print(f"running GLMsingle...")
    #sometimes the default 50,000 chunk length doesn't chunk into equal lengths, throwing an error when converting to array
    numvertices = data[0].shape[0]  # get shape of data for convenience
    opt['chunklen'] = int(numvertices) 

    glmsingle_obj = GLM_single(opt)
    # run GLMsingle
    glmsingle_obj.fit(
        design,
        data,
        stimDur,
        TR,
        outputdir=outputdir_glmsingle)
    elapsed_time = time.time() - start_time

    if args.verbose:
        print(
            '\telapsed time: ',
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
        )
    #save design matrix. save to outputdir_glmsingle after fitting glm, otherwise contents will be overwritten
    if args.verbose:
        print("saving design matrix")

    with open(os.path.join(outputdir_glmsingle, f"{subject}_{session}_task-action_conditionOrderDM.pkl"), 'wb') as f:
        pickle.dump((events_run, ses_conds), f)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"HumanActionsDataset") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-10 that you wish to process")
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
