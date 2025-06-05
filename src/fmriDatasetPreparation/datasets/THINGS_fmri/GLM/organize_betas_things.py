import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import argparse
import pickle
import glob as glob
import pandas as pd
from pathlib import Path

def main(args):
    #betas are zscored across sessions and compiled into a cross-session matrix for easy use. missing reps (in subjects 3, 4, 6, 8) are all nans
    fmri_path = os.path.join(args.dataset_root,"derivatives", "GLM")
    
    #the list of lists shows which sessions were grouped together to force repeated 
    # stimuli together to take advantage of glmsingle cross-validation procedures (type c and type d betas)
    sessiongroup_info = {"sub-01": [[1,2,3,4,5,6],[7,8,9,10,11,12]],
                      "sub-02": [[1,2,3,4,5,6],[7,8,9,10,11,12]],
                      "sub-03": [[1,2,3,4,5,6],[7,8,9,10,11,12]]
                      }  

    task = 'things' 
    subject = f"sub-{int(args.subject):02}"
    session_paths = glob.glob(os.path.join(fmri_path, subject, 'sessiongroup-*'))
    assert(len(session_paths) > 0)

    #divide the stimuli into test and train
    tmp = pd.read_table(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "things_stiminfo.tsv"),index_col=False)
    test_stimuli = tmp[(tmp['test_train'] == 'test') & (tmp[f"{subject}_reps"] > 0)]['filename'].tolist()
    train_stimuli = tmp[(tmp['test_train'] == 'train') & (tmp[f"{subject}_reps"] > 0)]['filename'].tolist()

    print(f"Length of test stim: {len(test_stimuli)}")
    print(f"Length of train stim: {len(train_stimuli)}")
    assert(len(test_stimuli) == 100)
    assert(len(train_stimuli) == 8640)

    betas_tmp_train = {stim: [] for stim in train_stimuli}
    betas_tmp_test = {stim: [] for stim in test_stimuli}
        
    #load fMRI data from that subject and run
    numsessiongroups = len(sessiongroup_info[subject])
    for ses in range(1,numsessiongroups+1):
        session = f"sessiongroup-{ses:02}"
        print(f"running session {session}")
        fmri_data_wb = np.load(os.path.join(fmri_path, subject, session, task, "TYPED_FITHRF_GLMDENOISE_RR.npy"), allow_pickle=True).item()
        fmri_data_wb = fmri_data_wb['betasmd'].squeeze() #squeezed to shape numvertices x numtrials
        fmri_data_wb = fmri_data_wb.T #transpose to shape numtrials x numvertices, more representative of the samples x features format    
        print(f"shape of betas in the session (numtrials x numvertices): {fmri_data_wb.shape}")

        with open(os.path.join(fmri_path, subject, session, task, f"{subject}_{session}_task-{task}_conditionOrderDM.pkl"), 'rb') as f:
            events_run, _ = pickle.load(f)
        ses_conds = []
        for event in events_run:
            trials = event['trial_type']
            for trial in trials:
                ses_conds.append(Path(trial).name) #the order the stimuli were presented

        test_idx = np.isin(ses_conds, test_stimuli)

        fmri_data_wb_train = fmri_data_wb[~test_idx, :]
        fmri_data_wb_test = fmri_data_wb[test_idx, :]

        #normalize the test and train betas by the train statistics
        train_mean = np.mean(fmri_data_wb_train, axis=0)
        train_std = np.std(fmri_data_wb_train, axis=0, ddof=1)

        fmri_data_wb_train_normalized = (fmri_data_wb_train - train_mean) / train_std
        fmri_data_wb_test_normalized = (fmri_data_wb_test - train_mean) / train_std

        train_count = 0
        test_count = 0
        for stim in ses_conds:
            if stim in test_stimuli:
                betas_tmp_test[stim].append(fmri_data_wb_test_normalized[test_count,:])
                test_count += 1
            elif stim in train_stimuli:
                betas_tmp_train[stim].append(fmri_data_wb_train_normalized[train_count, :])
                train_count += 1
        assert(test_count == fmri_data_wb_test_normalized.shape[0])
        assert(train_count == fmri_data_wb_train_normalized.shape[0])    

    numvertices = 91282
    numreps_train = 1
    numreps_test = 12
    betas_train = np.zeros((len(betas_tmp_train), numreps_train, numvertices))
    betas_test = np.zeros((len(betas_tmp_test), numreps_test, numvertices))
    betas_train.fill(np.nan)
    betas_test.fill(np.nan)
    stimorder_train = []
    stimorder_test = []
    for stimcount, b in enumerate(betas_tmp_train.keys()):
        value = betas_tmp_train[b]
        stimorder_train.append(b)
        for repcount, v in enumerate(value): #loop over reps
            betas_train[stimcount, repcount, :] = np.array(v)
    for stimcount, b in enumerate(betas_tmp_test.keys()):
        value = betas_tmp_test[b]
        stimorder_test.append(b)
        for repcount, v in enumerate(value): #loop over reps
            betas_test[stimcount, repcount, :] = np.array(v)

    #save betas
    save_root = os.path.join(fmri_path, f"{subject}", "prepared_betas")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print(f"saving {subject} train betas")
    with open(os.path.join(save_root, f"{subject}_organized_betas_task-train_normalized.pkl"), 'wb') as f:
        pickle.dump((betas_train,stimorder_train), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saving {subject} test betas")
    with open(os.path.join(save_root, f"{subject}_organized_betas_task-test_normalized.pkl"), 'wb') as f:
        pickle.dump((betas_test,stimorder_test), f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"THINGS_fmri") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-3 that you wish to process")
    parser.add_argument("-d", "--dataset_root",  default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
    args = parser.parse_args()
    
    main(args)