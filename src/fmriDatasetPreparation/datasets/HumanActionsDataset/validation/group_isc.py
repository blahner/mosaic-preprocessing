from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import argparse
import pickle
from scipy.stats import pearsonr
import pandas as pd

"""
Perform an intersubject correlation analysis on the categories. Each individuals betas are averaged over the 4 reps per category
to get a series of 180 beta values per vertex. The ISC is a leave-one-out correlation between a subject and the remaining 29 subject
group average. The final group ISC is the average of all the leave-one-out correlations. This is to replicate the results
of the HAD manuscript Figure 6 and validate our own preprocessing pipeline.
"""

def main(args):
    fmri_path = os.path.join(args.dataset_root,"derivatives", "GLM")

    #first define the stimulus order and matrix.
    #Next we will essentially place the betas into this pre-defined matrix
    df = pd.read_table(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "testtrain_split", "HAD_testtrain_split.tsv"),index_col=False)
    #get the order of categories
    categories = df['category'].unique()
    assert(len(categories) == 180)
    subject_betas = {} #this will be a big dictionary holding all the beta estimates from the subjects
    nvertices = 91282
    for sub in range(1,31):
        subject = f"sub-{sub:02}"
        #load the normalized betas for train and test
        with open(os.path.join(fmri_path, subject, "prepared_betas", f"{subject}_organized_betas_task-train_normalized.pkl"), 'wb') as f:
            betas_train, stimorder_train = pickle.load(f)
        with open(os.path.join(fmri_path, subject, "prepared_betas", f"{subject}_organized_betas_task-test_normalized.pkl"), 'wb') as f:
            betas_test, stimorder_test = pickle.load(f)

        #map the stimulus to categories in order
        categoryorder_train = []
        for stim in stimorder_train:
            tmp = stim.split('_id_')[-1]
            cat = tmp.split('v_')[-1]
            categoryorder_train.append(cat)
        categoryorder_test = []
        for stim in stimorder_test:
            tmp = stim.split('_id_')[-1]
            cat = tmp.split('v_')[-1]
            categoryorder_test.append(cat)

        #sort into categories
        beta_categories = np.zeros((len(categories), nvertices))
        for count, cat in enumerate(categories):
            cat_idx_train = np.isin(categoryorder_train, cat)
            cat_idx_test = np.isin(categoryorder_test, cat)
            assert(cat_idx_train.sum() == 3)
            assert(cat_idx_test.sum() == 1)

            betas_tmp = np.concatenate(betas_train[cat_idx_train,0,:], betas_test[cat_idx_test, :])
            assert(len(betas_tmp) == 4)

            beta_categories[count, :] = np.mean(betas_tmp)
        
        subject_betas.update({subject: beta_categories})

    isc = np.zeros((nvertices,))
    for left_out_subject in subject_betas.keys():
        left_out_betas = subject_betas[left_out_subject]
        remaining_betas = np.mean([value for key, value in subject_betas.items() if key != left_out_subject])
        isc += pearsonr(left_out_betas, remaining_betas)
    
    isc = isc/len(subject_betas) #average the individual isc

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"HumanActionsDataset") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root",  default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)