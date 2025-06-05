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

#local
from src.utils.helpers import interpolate_ts

#compute glm using glmsingle
# https://github.com/cvnlab/GLMsingle

def main(args):
    subject = f"sub-{int(args.subject):02}"
    assert(int(args.session) > 1 and int(args.session) < 6)
    session = f"ses-{int(args.session):02}"
    if args.verbose:
        print("Running GLMsingle on the main experiment task for subject {}".format(subject))

    TR = 1.75 #acquisition TR
    TR_resamp = 1 # resample time series to be locked to stimulus onset
    stimDur = 3 #in seconds
    dummy_offset = 0 #offset of start. in seconds

    if args.verbose:
        print("#" * 20)
        print("Starting GLMsingle on main experimental data for subject {} session {}".format(subject, session))
        print("#" * 20)
    sub_func_root = os.path.join(args.dataset_root, "derivatives", "versionC", "fmriprep", subject, session, "func")

    data = []
    design = []

    cols = ['trial_type','onset']
    events_run = []     
    ##Step X: Load eventts and data for each run
    ses_conds = [] #keep track of the test conditions shown in this session over all runs
    for task in ['test', 'train']:
        if task == 'test':
            numruns = 3
        elif task == 'train':
            numruns = 10
        for count, run in enumerate(range(1,numruns+1)):
            if args.verbose:
                print(f"task {task} run {run}")
            #load data
            cifti_ts = nib.load(os.path.join(sub_func_root, f"{subject}_{session}_task-{task}_run-{run}_space-fsLR_den-91k_bold.dtseries.nii"))
            cifti_data = cifti_ts.get_fdata()

            #interpolate time series
            fmri_interp = interpolate_ts(cifti_data.T, TR, TR_resamp)
            numscans_interp = fmri_interp.shape[1]
            data.append(fmri_interp)

            #load events
            events_tmp = {col: [] for col in cols}  
            tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run}_events.tsv"))
            for idx, tt in enumerate(tmp.loc[:,'stim_file']):
                if str(tt) == "nan":
                    continue
                else:
                    fname = tt.split(f"{task}/")[1]
                    stimIDX = int(fname.split(".mp4")[0]) #stimuli index 1001-1102
                    trial_type = "vid" + str(stimIDX).zfill(4)
                    if trial_type not in ses_conds:
                        ses_conds.append(trial_type)
                    onset = tmp.loc[idx,'onset']
                    events_tmp['trial_type'].append(trial_type)
                    events_tmp['onset'].append(onset + dummy_offset)
            events_run.append(events_tmp)

    #create the design matrix for all runs in the session
    for count, _ in enumerate(range(1,len(data)+1)):
        numscans_interp = data[count].shape[1]
        run_design = np.zeros((numscans_interp, len(ses_conds)))
        events = events_run[count]
        for c, cond in enumerate(ses_conds):
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

    outputdir_glmsingle = os.path.join(args.dataset_root, "derivatives", "versionC", "GLM", subject, session)
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
    with open(os.path.join(outputdir_glmsingle, f"{subject}_{session}_conditionOrderDM.pkl"), 'wb') as f:
        pickle.dump((events_run, ses_conds), f)

    #TODO figure out cifti files
    """
    #additionally save betas as cifti file with cifti header
    betas = np.load(os.path.join(outputdir_glmsingle, "TYPED_FITHRF_GLMDENOISE_RR.npy"), allow_pickle=True).item()
    bm = list(cifti.header.get_index_map(1).brain_models)[0:2] #uses the header information from the last loaded run
    scalar_axis = nib.cifti2.ScalarAxis(['betas_raw'])  # Takes a list of names, one per row
    new_header = nib.Cifti2Header.from_axes([scalar_axis, bm])
    save2cifti(file_path=os.path.join(outputdir_glmsingle, f"{subject}_betas-typed.dscalar.nii") , data=betas['betasmd'].T, brain_models=bm, map_names=ses_conds) #data must be in (maps, values)
    """

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"BOLDMomentsDataset") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-10 that you wish to process")
    parser.add_argument("-i", "--session", type=int, required=True, help="The session you want to perform glmsingle on.")
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
    args = parser.parse_args()

    main(args)