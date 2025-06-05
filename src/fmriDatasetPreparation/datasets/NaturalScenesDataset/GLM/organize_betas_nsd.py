from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import argparse
from pycocotools.coco import COCO

import pickle
import pandas as pd

def main(args):
    #betas are zscored across sessions and compiled into a cross-session matrix for easy use. missing reps (in subjects 3, 4, 6, 8) are all nans
    fmri_path = os.path.join(args.dataset_root,"derivatives", "GLM")
    session_info = {'sub-01': 40, 'sub-02': 40, 'sub-03': 32, 'sub-04': 30,
                'sub-05': 40, 'sub-06': 32, 'sub-07': 40, 'sub-08': 30}

    subject = f"sub-{int(args.subject):02}"
    print(f"starting {subject}")
    numsessions = session_info[subject]
    #load stimuli info
    annotations_root = os.path.join(args.dataset_root, "derivatives", "stimuli_metadata")
    nsd_csv = pd.read_csv(os.path.join(annotations_root, "nsd_stim_info_merged.csv"))
    coco_annotation_val = COCO(annotation_file=os.path.join(annotations_root, "annotations_trainval2017", "annotations", "instances_val2017.json"))
    coco_annotation_train = COCO(annotation_file=os.path.join(annotations_root, "annotations_trainval2017", "annotations", "instances_train2017.json"))       
    img_ids_val = coco_annotation_val.getImgIds()
    img_ids_train = coco_annotation_train.getImgIds()   

    #divide the stimuli into test and train
    tmp = pd.read_table(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "nsd_stiminfo.tsv"),index_col=False)
    test_stimuli = tmp[(tmp['test_train'] == 'test') & (tmp[f"{subject}_reps"] > 0) & (tmp["source"] == 'COCO')]['filename'].tolist()
    train_stimuli = tmp[(tmp['test_train'] == 'train') & (tmp[f"{subject}_reps"] > 0) & (tmp["source"] == 'COCO')]['filename'].tolist()
    
    print(f"Length of test stim: {len(test_stimuli)}")
    print(f"Length of train stim: {len(train_stimuli)}")
    
    betas_tmp_train = {stim: [] for stim in train_stimuli}
    betas_tmp_test = {stim: [] for stim in test_stimuli}

    #load fMRI data from that subject and run
    for ses in range(1,numsessions+1):
        print(f"loading session {ses}")
        session = f"ses-nsd{ses:02}"
        fmri_data_wb = np.load(os.path.join(fmri_path, subject, session, "TYPED_FITHRF_GLMDENOISE_RR.npy"), allow_pickle=True).item()
        fmri_data_wb = fmri_data_wb['betasmd'].squeeze() #squeezed to shape numvertices x numtrials
        fmri_data_wb = fmri_data_wb.T #transpose to shape numtrials x numvertices, more representative of the samples x features format
        with open(os.path.join(fmri_path, subject, session, f"{subject}_{session}_task-nsdcore_conditionOrderDM.pkl"), 'rb') as f:
            events_run, _ = pickle.load(f)
        ses_conds = []
        for event in events_run:
            trial_all = event['trial_type']
            for t in trial_all:
                trial_nsdId = int(t.split('73k_id-')[-1]) - 1 
                #nsd 73k id to coco id
                cocoID = nsd_csv.loc[nsd_csv['nsdId'] == (trial_nsdId), 'cocoId'].to_list()[0] 
                #coco id to image filename
                if cocoID in img_ids_val:
                    img_info = coco_annotation_val.loadImgs(cocoID)[0]
                elif cocoID in img_ids_train:
                    img_info = coco_annotation_train.loadImgs(cocoID)[0]
                else:
                    raise ValueError(f"cocoID {cocoID} not found in test or train info.")
                ses_conds.append(img_info['file_name'])

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
                betas_tmp_test[stim].append(fmri_data_wb_test_normalized[test_count, :])#+1 to the 73k-id to put it back intot the 1-indexed form of the events file
                test_count += 1
            elif stim in train_stimuli:
                betas_tmp_train[stim].append(fmri_data_wb_train_normalized[train_count, :])
                train_count += 1
            else:
                raise ValueError(f"stimulus {stim} not found in test or train set")

    numvertices = 91282
    numreps = 3
    betas_train = np.zeros((len(betas_tmp_train), numreps, numvertices))
    betas_test = np.zeros((len(betas_tmp_test), numreps, numvertices))
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
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"NaturalScenesDataset") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-8 that you wish to process")
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
    args = parser.parse_args()

    main(args)