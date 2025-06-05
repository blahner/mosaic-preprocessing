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
parser.add_argument("-g", "--glm_root", default="/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset", help="The root path to the GLM analysis")
parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
parser.add_argument("-z", "--zscore", action="store_true", help="bool to zscore (true) or not")
parser.add_argument("-b", "--beta_type", default='typeb', help="Which GLMsingle beta output you want to analyze. [typea, typeb, typec, typed]")

args = parser.parse_args()
"""
class arguments:
    def __init__(self) -> None:
        self.subject = "1"         
        self.task = "train"
        self.glm_root = "/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset"
        self.verbose =True
        self.beta_type = "typeb"
        self.zscore = True
args = arguments()
"""
assert(args.beta_type in ['typea', 'typeb', 'typec', 'typed'])
assert(args.task in ['test','train'])
subject = f"subject{int(args.subject)}"
if args.verbose:
    print(f"Preparing raw betas from the {args.task} task for subject {args.subject}")

beta_types = {"typea": "TYPEA_ONOFF", "typeb": "TYPEB_FITHRF" , "typec": "TYPEC_FITHRF_GLMDENOISE", "typed": "TYPED_FITHRF_GLMDENOISE_RR"}
beta_type = beta_types[args.beta_type]

sub_save_root = os.path.join(args.glm_root, "GLMsingleAllRuns", subject, "betas-prepared", "step01")
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
    print("Organizing betas into matrix")
betas = [] #should be len 8604 x 91282 for train
condition_order = []
for seg in range(1,numsegs+1):
    sub_beta_root = os.path.join(args.glm_root, "GLMsingleAllRuns", subject, args.task, f"{prefix}{seg}")
    #load the betas
    betas_tmp = np.load(os.path.join(sub_beta_root, beta_type + ".npy"), allow_pickle=True).item()
    betas_tmp = betas_tmp['betasmd'].squeeze()
    #load conditions
    conds_tmp1 = np.load(os.path.join(sub_beta_root, f"{subject}_{prefix}{seg}_stimOrder.npy"))
    conds_tmp2 = []
    for c in conds_tmp1:
        conds_tmp2.extend(c)
    assert(len(conds_tmp2) == betas_tmp.shape[1])
    for idx, c in enumerate(conds_tmp2):
        condition_order.append(c)
        betas.append(betas_tmp[:,idx])

nvertices = betas[0].shape[0]

betas_matrix = np.stack(betas)

assert(betas_matrix.shape[1] == nvertices) #just to make sure dimension order is correct
if args.verbose:   
    print("shape of beta matrix (#stim, #vertices): {}".format(betas_matrix.shape))
    #save the betas in nice format
    print("Saving raw beta matrix")

beta_filepath_raw = os.path.join(sub_save_root, f"{subject}_z=0_GLMsingle_type-{args.beta_type}_task-{args.task}.pkl")
with open(beta_filepath_raw,'wb') as f:
    pickle.dump((betas_matrix, condition_order), f)

#optionally zscore betas
if args.zscore:
    if args.verbose:
        print("zscoring data for subject {}".format(subject))
    z_string = 'z=1'
    #zscoring across videos make the response profile at each vertex and each repetition a mean of 0 and std of 1
    betas_matrix_z = stats.zscore(betas_matrix, axis=0, ddof=1, nan_policy='propagate') #axis=0 zscores across videos
    beta_filepath_z = os.path.join(sub_save_root, f"{subject}_{z_string}_GLMsingle_type-{args.beta_type}_task-{args.task}.pkl")

    #save both raw and zscored betas
    with open(beta_filepath_z,'wb') as f:
        pickle.dump((betas_matrix_z, condition_order), f)