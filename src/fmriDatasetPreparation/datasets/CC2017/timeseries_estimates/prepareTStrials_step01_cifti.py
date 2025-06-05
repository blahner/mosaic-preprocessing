# Standard library imports
from dotenv import load_dotenv
load_dotenv()
import os
import warnings
warnings.filterwarnings('ignore')
import pickle
import argparse

# Third party
import numpy as np
import nibabel as nib
from scipy import stats

"""
Step 1 of 2 in preparing the beta estimates for analyses.
Converts the raw beta estimates into a format easy to use for 
analyses i.e. voxels x reps x vids.
Works for testing and training tasks
"""
def main(args):
    assert(args.task in ['test','train'])
    subject = f"subject{int(args.subject)}"
    if args.verbose:
        print("Preparing raw trial estimates from the {} task for subject {}".format(args.task, subject))

    sub_save_root = os.path.join(args.root, "video_fmri_dataset", "TSTrialEstimates", subject, "estimates-prepared", "step01")
    if not os.path.exists(sub_save_root):
        os.makedirs(sub_save_root)

    if args.task == "test":
        numsegs = 5
        numreps = 10
        prefix = "test"
    elif args.task == "train":
        numsegs = 18
        numreps = 2
        prefix = "seg"
    if args.verbose:
        print("Organizing trial estimates into matrix")

    estimates = [] #should be len 8604 x 91282 for train
    condition_order = []
    for seg in range(1,numsegs+1):
        for rep in range(1,numreps+1):
            if args.verbose:
                print(f"preprocessing data for {subject} segment {seg} repetition {rep}")
            sub_ts_root = os.path.join(args.root, "video_fmri_dataset", "TSTrialEstimates", subject, args.task, f"{prefix}{seg}_{rep}")
            #load the trial estimates
            estimates_tmp = np.load(os.path.join(sub_ts_root, f"{subject}_{prefix}{seg}_rep-{rep}_estimates.npy"))
            if args.zscore:
                #zscoring across videos make the response profile at each vertex and each repetition a mean of 0 and std of 1
                estimates_tmp = stats.zscore(estimates_tmp, axis=0, ddof=1, nan_policy='propagate') #axis=0 zscores across videos

            #load conditions
            conds = np.load(os.path.join(sub_ts_root, f"{subject}_{prefix}{seg}_rep-{rep}_stimOrder.npy"))
            
            assert(len(conds[0]) == estimates_tmp.shape[0])
            for idx, cond in enumerate(conds[0]):
                estimates.append(estimates_tmp[idx,:])
                condition_order.append(cond)

    estimates_matrix = np.array(estimates)
    assert(estimates_matrix.shape[1] == 91282) #just to make sure dimension order is correct

    if args.zscore:
        z_string = 'z=1' #z=1 denotes that the values are zscored
    else:
        z_string = 'z=0' #z=0 denotes that the values are not zscored

    estimates_filepath = os.path.join(sub_save_root, f"{subject}_{z_string}_TSTrialEstimates_task-{args.task}.pkl")
    #save both raw and zscored estimates
    with open(estimates_filepath,'wb') as f:
        pickle.dump((estimates_matrix, condition_order), f)

if __name__=='__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "CC2017") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-3 that you wish to process")
    parser.add_argument("-t", "--task", required=True, help="Which task to analyze, either test or train")
    parser.add_argument("-g", "--root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
    parser.add_argument("-z", "--zscore", action="store_true", help="bool to zscore (true) or not")

    args = parser.parse_args()

    main(args)