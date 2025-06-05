from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import argparse
import pickle
import pandas as pd

#the test set is the last three runs of each session. These last three runs encompass a rep from each of the 180 categories
def main(args):
    subject = f"sub-{int(args.subject):02}"
    print("*"*20)
    print(f"starting subject {subject}")
    print("*"*20)
    session = "ses-action01"    

    #divide the stimuli into test and train
    tmp = pd.read_table(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "had_stiminfo.tsv"),index_col=False)
    test_stimuli = tmp[(tmp['test_train'] == 'test') & (tmp[f"{subject}_reps"] > 0)]['filename'].tolist()
    train_stimuli = tmp[(tmp['test_train'] == 'train') & (tmp[f"{subject}_reps"] > 0)]['filename'].tolist()

    print(f"Length of test stim: {len(test_stimuli)}")
    print(f"Length of train stim: {len(train_stimuli)}")
    assert(len(test_stimuli) == 180)
    assert(len(train_stimuli) == 540)
    assert(len(set(test_stimuli) & set(train_stimuli)) == 0) #no overlap between train and test

    betas_tmp_train = {stim: [] for stim in train_stimuli}
    betas_tmp_test = {stim: [] for stim in test_stimuli}

    fmri_path = os.path.join(args.dataset_root,"derivatives", "GLM", subject)

    print(f"starting {subject}")
    #load fMRI data from that subject and ROI
    fmri_data_wb = np.load(os.path.join(fmri_path, session, "TYPEB_FITHRF.npy"), allow_pickle=True).item()
    fmri_data_wb = fmri_data_wb['betasmd'].squeeze() #squeezed to shape numvertices x numtrials
    fmri_data_wb = fmri_data_wb.T #transpose to shape numtrials x numvertices, more representative of the samples x features format  
    print(f"shape of betas in the session (numtrials x numvertices): {fmri_data_wb.shape}")
  
    with open(os.path.join(fmri_path, session, f"{subject}_{session}_task-action_conditionOrderDM.pkl"), 'rb') as f:
        events_run, _ = pickle.load(f)
    ses_conds = []
    for event in events_run:
        trials = event['trial_type']
        for trial in trials:
            ses_conds.append(f"{trial}.mp4") #the order the stimuli were presented

    test_idx = np.isin(ses_conds, test_stimuli)

    fmri_data_wb_train = fmri_data_wb[~test_idx, :]
    fmri_data_wb_test = fmri_data_wb[test_idx, :]

    #normalize the test and train betas by the train statistics
    train_mean = np.mean(fmri_data_wb_train, axis=0)
    train_std = np.std(fmri_data_wb_train, axis=0, ddof=1)

    zero_val_mean = np.any(train_mean==0)
    zero_val_std = np.any(train_std==0)
    if zero_val_mean or zero_val_std:
        #make sure this number isnt too big. A handful of veritices is fine, but if this number is into
        #the thousands that may be indicative of a larger problem. I saw subs 1 and 12 have 7 vertices with mean and std of 0
        print(f"Warning: found {(train_mean==0).sum()} vertices with a mean of 0")
        print(f"Warning: found {(train_std==0).sum()} vertices with a std of 0")

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

    numreps = 1
    numvertices = 91282

    betas_train = np.zeros((len(betas_tmp_train), numreps, numvertices))
    betas_train.fill(np.nan)
    betas_test = np.zeros((len(betas_tmp_test), numreps, numvertices))
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
    save_root = os.path.join(fmri_path, "prepared_betas")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print(f"saving {subject} train betas")
    with open(os.path.join(save_root, f"{subject}_organized_betas_task-train_normalized.pkl"), 'wb') as f:
        pickle.dump((betas_train,stimorder_train), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saving {subject} test betas")
    with open(os.path.join(save_root, f"{subject}_organized_betas_task-test_normalized.pkl"), 'wb') as f:
        pickle.dump((betas_test,stimorder_test), f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"HumanActionsDataset") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-30 that you wish to process")
    parser.add_argument("-d", "--dataset_root",  default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)