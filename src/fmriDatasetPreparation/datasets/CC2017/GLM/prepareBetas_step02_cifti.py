#standard
import os
import warnings
warnings.filterwarnings('ignore')
import argparse
import pickle

#third party
import nibabel as nib
import numpy as np
import scipy
from nilearn import plotting
import hcp_utils as hcp
import pandas as pd

def calculate_noiseceiling(betas):
    """
    Calculate the standard deviation across trials, square the result,
    average across images, and then take the square root. The result is
    the estimate of the 'noise standard deviation'. This is done for the training and
    testing sets separately because of the different number of stimuli repetitions.
    Parameters:
    betas: beta estimates in shape (vertices, num_reps, num_stimuli)
    Returns:
    ncsnr: noise-ceiling SNR at each voxel in shape (voxel_x, voxel_y, voxel_z) as ratio between signal std and noise std
    noiseceiling: noise ceiling at each voxel in shape (voxel_x, voxel_y, voxel_z) as % of explainable variance 
    Code adapted from GLMsingle example: https://github.com/cvnlab/GLMsingle/blob/main/examples/example9_noiseceiling.ipynb
    """
    assert(len(betas.shape) == 3)
    numvertices = betas.shape[0]
    num_reps = betas.shape[1]
    num_vids = betas.shape[2]
    noisesd = np.sqrt(np.mean(np.power(np.std(betas,axis=1,keepdims=1,ddof=1),2),axis=2)).reshape((numvertices,))

    # Calculate the total variance of the single-trial betas.
    totalvar = np.power(np.std(np.reshape(betas, (numvertices , num_reps*num_vids)), axis=1),2)

    # Estimate the signal variance and positively rectify.
    signalvar = totalvar - np.power(noisesd,2)

    signalvar[signalvar < 0] = 0
    # Compute ncsnr as the ratio between signal standard deviation and noise standard deviation.
    ncsnr = np.sqrt(signalvar) / noisesd

    # Compute noise ceiling in units of percentage of explainable variance
    # for the case of 3 trials.
    noiseceiling = 100 * (np.power(ncsnr,2) / (np.power(ncsnr,2) + 1/num_reps))
    return ncsnr, noiseceiling

def list_rep(myList: list, reps: int):
    #returns a list of items in "mylist" that are repeated "reps" number of times
    repList = []
    # traverse for all elements
    for x in myList:
        if x not in repList: 
            count = myList.count(x)
            if count == reps:
                repList.append(x)
    return repList

"""
Step 2 of 3 in preparing the beta estimates for analyses.
Removes nans and zscores (across conditions) the raw reformatted beta estimates.
If a voxel is nan in either the testing or training runs, then it is nan for both.
If zscore is True, betas are zscored across conditions for training and testing data
separately.
Needs Step 01 to be run.
"""

#arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-10 that you wish to process")
parser.add_argument("-g", "--glm_root", default="/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset", help="The root path to the GLM analysis")
parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
parser.add_argument("-b", "--beta_type", default='typeb', help="Which GLMsingle beta output you want to analyze. [typea, typeb, typec, typed]")
parser.add_argument("-z", "--zscore", action="store_true", help="Bool to zscore the betas across conditions or not")
parser.add_argument("-p", "--plot", action="store_true", help="Bool to plot noise ceiling")

args = parser.parse_args()
"""
class arguments:
    def __init__(self) -> None:
        self.subject = "1"         
        self.glm_root = "/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset"
        self.verbose =True
        self.beta_type = "typeb"
        self.zscore = True
        self.plot=True
args = arguments()
"""
assert(args.beta_type in ['typea', 'typeb', 'typec', 'typed'])
subject = f"subject{int(args.subject)}"
views = ['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior']
beta_types = {"typea": "TYPEA_ONOFF", "typeb": "TYPEB_FITHRF" , "typec": "TYPEC_FITHRF_GLMDENOISE", "typed": "TYPED_FITHRF_GLMDENOISE_RR"}
beta_type = beta_types[args.beta_type]

sub_save_root = os.path.join(args.glm_root, "GLMsingleAllRuns", subject, "betas-prepared", "prepared_allvoxel_pkl")
if not os.path.exists(sub_save_root):
    os.makedirs(sub_save_root)
step_save_root = os.path.join(args.glm_root,"GLMsingleAllRuns", subject, "betas-prepared", "step02")
if not os.path.exists(step_save_root):
    os.makedirs(step_save_root)

if args.zscore:
    z_string = 'z=1'
else:
    z_string = 'z=0'

if args.verbose:
    print("defining ROI groups")
groups = list(np.arange(1,23)) #[1,2,3,4,5,6,7,8,9,15,16,17,18] #groups from the glasser parcellation I want. see table 1 in glasser supplementary for details
tmp = pd.read_table(os.path.join(args.glm_root, "utils","hcp_glasser_roilist.txt"), sep=',')
roi_idx_running = {}
for count, li in enumerate(range(tmp.shape[0])):
    line = tmp.iloc[count,:]
    ROI = line['ROI']
    GROUP = line['GROUP']
    ID = line['ID']
    if GROUP in groups: #if the roi is in a group we want, include that roi
        roi_idx_running[ROI] = np.where(((hcp.mmp.map_all == ID)) | (hcp.mmp.map_all == ID+180))[0]

#add a group ROI
group_indices = []
with open(os.path.join(args.glm_root, "utils","roi_list_reduced41.txt"), 'r') as f:
    tmp = f.read().splitlines()
for roi in tmp:
    roi = roi[1:-1]
    group_indices.extend(roi_idx_running[roi])
roi_idx_running["Group41"] = np.array(group_indices)

if args.verbose:
    print("loading non-zscored betas for noiseceiling calculation")
#load non-zscored betas for noisceiling calculation
#compute noise ceiling on non-zscored betas
with open(os.path.join(args.glm_root, "GLMsingleAllRuns", subject, "betas-prepared", "step01", f"{subject}_z=0_GLMsingle_type-{args.beta_type}_task-test.pkl"), 'rb') as f:
    betas_noz_test, condition_order_test = pickle.load(f)

if args.verbose:
    print("Aggregating repeated conditions")
numvertices = betas_noz_test.shape[1]
numreps = 10
repeated_conditions = list_rep(condition_order_test, 10) #just get the conditions repeated exactly numpreps times
betas_noz_test_matrix = np.zeros((numvertices, numreps, len(repeated_conditions)))
for count, cond in enumerate(repeated_conditions):
    idx = [i for i, c in enumerate(condition_order_test) if c == cond]
    betas_noz_test_matrix[:,:,count] = betas_noz_test[idx, :].T

if args.verbose:
    print("Computing noiseceiling")
#reformat betas for matrix for noisceiling
ncsnr_test, noiseceiling_test = calculate_noiseceiling(betas_noz_test_matrix)
if args.verbose:
    print(f"max noiseceiling: {noiseceiling_test.max()}")

#save noise ceiling on non-zscored betas
if args.verbose:
    print("saving noiseceilings...")
with open(os.path.join(step_save_root, f"{subject}_noiseceiling_task-test.pkl"), 'wb') as f:
    pickle.dump((ncsnr_test, noiseceiling_test), f)

if args.verbose:
    print("Deleting large variables to free room")
del betas_noz_test, betas_noz_test_matrix #free RAM

if args.verbose:
    print("Loading zscored betas")
#load zscored betas
with open(os.path.join(args.glm_root, "GLMsingleAllRuns", subject, "betas-prepared", "step01", f"{subject}_{z_string}_GLMsingle_type-{args.beta_type}_task-train.pkl"), 'rb') as f:
    betas_matrix_train, condition_order_train = pickle.load(f)
with open(os.path.join(args.glm_root, "GLMsingleAllRuns", subject, "betas-prepared", "step01", f"{subject}_{z_string}_GLMsingle_type-{args.beta_type}_task-test.pkl"), 'rb') as f:
    betas_matrix_test, condition_order_test = pickle.load(f)

if args.verbose:
    print("Saving ROI data...")
for roi, roi_indices in roi_idx_running.items():
    fmri_train = betas_matrix_train[:, roi_indices]
    fmri_test = betas_matrix_test[:, roi_indices]
    data = {}
    data['train_data_allvoxel'] = fmri_train #(#videos, #reps, #grayordinates)
    data['test_data_allvoxel'] = fmri_test #(#videos, #reps, #grayordinates)
    
    data['train_stim_order'] = condition_order_train
    data['test_stim_order'] = condition_order_test

    data['test_noiseceiling_allvoxel'] = noiseceiling_test #(#grayordinates)

    data['roi_indices_hcp'] = roi_indices
    with open(os.path.join(sub_save_root, f"{roi}_betas-GLMsingle_type-{args.beta_type}_{z_string}.pkl"), 'wb') as f:
        pickle.dump(data,f)

if args.plot:
    if args.verbose:
        print("plotting noiseceling results")
    plot_root = os.path.join(step_save_root, "plots")
    if not os.path.exists(plot_root):
        os.makedirs(plot_root)
    task = 'test'
    stat = noiseceiling_test.copy()
    for hemi in ['left','right']:
        for view in views:
            plotting.plot_surf_stat_map(hcp.mesh.inflated, hcp.cortex_data(stat), hemi=hemi,
            threshold=1.5, bg_map=hcp.mesh.sulc, view=view,
            output_file=os.path.join(plot_root, f"{subject}_noiseceiling_task-{task}_hemi-{hemi}_view-{view}.png"))