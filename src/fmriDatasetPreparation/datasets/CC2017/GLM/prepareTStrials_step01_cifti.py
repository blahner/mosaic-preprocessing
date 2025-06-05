# Standard library imports
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
Step 1 of 3 in preparing the beta estimates for analyses.
Converts the raw beta estimates into a format easy to use for 
analyses i.e. voxels x reps x vids.
Works for testing and training tasks
"""

#arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-10 that you wish to process")
parser.add_argument("-t", "--task", required=True, help="Which task to analyze, either test or train")
parser.add_argument("-g", "--root", default="/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset", help="The root path to the GLM analysis")
parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
parser.add_argument("-z", "--zscore", action="store_true", help="bool to zscore (true) or not")

args = parser.parse_args()
"""
class arguments:
    def __init__(self) -> None:
        self.subject = "1"         
        self.task = "train"
        self.root = "/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset"
        self.verbose =True
        self.zscore = True
args = arguments()
"""
assert(args.task in ['test','train'])
subject = f"subject{int(args.subject)}"
if args.verbose:
    print("Preparing raw trial estimates from the {} task for subject {}".format(args.task, subject))

sub_save_root = os.path.join(args.root, "TSTrialEstimates", subject, "estimates-prepared", "step01")
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
        sub_ts_root = os.path.join(args.root, "TSTrialEstimates", subject, args.task, f"{prefix}{seg}_{rep}")
        #load the trial estimates
        estimates_tmp = np.load(os.path.join(sub_ts_root, f"{subject}_{prefix}{seg}_rep-{rep}_estimates.npy"))
        #load conditions
        conds = np.load(os.path.join(sub_ts_root, f"{subject}_{prefix}{seg}_rep-{rep}_stimOrder.npy"))
        
        assert(len(conds[0]) == estimates_tmp.shape[0])
        for idx, cond in enumerate(conds[0]):
            estimates.append(estimates_tmp[idx,:])
            condition_order.append(cond)

estimates_matrix = np.array(estimates)
assert(estimates_matrix.shape[1] == 91282) #just to make sure dimension order is correct
if args.verbose:   
    print("shape of estimates matrix (#stim, #vertices): {}".format(estimates_matrix.shape))
    #save the estimates in nice format
    print("Saving raw estimates matrix")

estimates_filepath_raw = os.path.join(sub_save_root, f"{subject}_z=0_TSTrialEstimates_task-{args.task}.pkl")
with open(estimates_filepath_raw,'wb') as f:
    pickle.dump((estimates_matrix, condition_order), f)

#optionally zscore estimates
if args.zscore:
    if args.verbose:
        print("zscoring data for subject {}".format(subject))
    z_string = 'z=1'
    #zscoring across videos make the response profile at each vertex and each repetition a mean of 0 and std of 1
    estimates_matrix_z = stats.zscore(estimates_matrix, axis=0, ddof=1, nan_policy='propagate') #axis=0 zscores across videos
    estimates_filepath_z = os.path.join(sub_save_root, f"{subject}_{z_string}_TSTrialEstimates_task-{args.task}.pkl")

    #save both raw and zscored estimates
    with open(estimates_filepath_z,'wb') as f:
        pickle.dump((estimates_matrix_z, condition_order), f)