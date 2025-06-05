from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import argparse
import pickle
from scipy import stats
import glob as glob
import pandas as pd

def main(args):
    fmri_path = os.path.join(args.dataset_root,"derivatives", "GLM")
    test_map = pd.read_csv(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "images", "image_test_id.csv"), names=['stim_id', 'filename'])
    train_map = pd.read_csv(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "images", "image_training_id.csv"), names=['stim_id', 'filename'])   
    subject = f"sub-{int(args.subject):02}"

    #divide the stimuli into test and train
    tmp = pd.read_table(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "god_stiminfo.tsv"),index_col=False)
    test_stimuli = tmp[(tmp['test_train'] == 'test') & (tmp[f"{subject}_reps"] > 0)]['filename'].tolist()
    train_stimuli = tmp[(tmp['test_train'] == 'train') & (tmp[f"{subject}_reps"] > 0)]['filename'].tolist()

    print(f"Length of test stim: {len(test_stimuli)}")
    print(f"Length of train stim: {len(train_stimuli)}")
    assert not set(test_stimuli) & set(train_stimuli), "Error: Train and Test stimuli lists overlap"
    assert(len(test_stimuli) == 50)
    assert(len(train_stimuli) == 1200)

    betas_tmp_train = {stim: [] for stim in train_stimuli}
    betas_tmp_test = {stim: [] for stim in test_stimuli}

    session_group = f"sessiongroup-01" #for each subject there will only be one session group.

    print(f"loading session group {session_group}")
    fmri_data_wb = np.load(os.path.join(fmri_path, subject, session_group, "TYPED_FITHRF_GLMDENOISE_RR.npy"), allow_pickle=True).item()
    fmri_data_wb = fmri_data_wb['betasmd'].squeeze() #squeezed to shape numvertices x numtrials
    fmri_data_wb = fmri_data_wb.T #transpose to shape numtrials x numvertices, more representative of the samples x features format    
    
    with open(os.path.join(fmri_path, subject, session_group, f"{subject}_{session_group}_task-perception_conditionOrderDM.pkl"), 'rb') as f:
        events_run, _ = pickle.load(f)
    ses_conds = []
    for event in events_run:
        for stim_id in event['trial_type']:
            if stim_id in train_map['stim_id'].values:
                ses_conds.append(train_map.loc[train_map['stim_id'] == stim_id, 'filename'].to_list()[0])
            elif stim_id in test_map['stim_id'].values:
                ses_conds.append(test_map.loc[test_map['stim_id'] == stim_id, 'filename'].to_list()[0])
            else:
                raise ValueError(f"stim_id {stim_id} not found in test or train sets.")

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
    numreps_test = 35

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
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"GenericObjectDecoding") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-5 that you wish to process")
    parser.add_argument("-d", "--dataset_root",  default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
    args = parser.parse_args()

    main(args)