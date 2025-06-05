import numpy as np
import scipy
import copy
import os
from os.path import join, exists
import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from nilearn import surface
import numpy as np
import nilearn
import os
import scipy.io as sio
import hcp_utils as hcp
import argparse
import nibabel as nib
from scipy.stats import pearsonr
from nilearn import plotting
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

#look at wen's README for some GLM parameters and warnings
"""
Each experiment included multiple sessions of 8 min and 24 s long. 
During each session, an 8-min single movie segment was presented; 
before the movie presentation, the first movie frame was displayed
 as a static picture for 12 s; after the movie, the last movie 
 frame was also displayed as a static picture for 12 s.

 Take caution:
1. During the experiments for all the training movie segments and the first testing movie 
segment (i.e. test1.mp4), the fMRI data recording started right after the movie were 
presented for 2 seconds, but ended at the same time as movie ended (see the figure bellow).
For all the other four testing movie segments (test2, test3, test4 and test5), the fMRI 
recording and stimuli presenting started and ended at the same time. Please pay attention 
to the number of volumes when processing the fMRI data.
2. For Subject3, during one fMRI scanning session (i.e. session "test2_9"), the fMRI recording
was around 10s behind the movie "test2.mp4" playing. Therefore, take caution when using 
"test2_9.nii.gz", "test2_9_mni.nii.gz", and "test2_9_Atlas.dtseries.nii"

Ben's comments:
The picture in the README does not match the text description and number of volumes acquired in the
dtseries files. I think fmri scanner began acquisition at 14s after the researcher pressed play (after
the 12s of static first frame and 2s of movie) and ended appropriately, after the 12s of static last frame.

I'm assuming "10 seconds behind the movie" for subject 3 test2 rep 9 means the fMRI started during second 2 of
the static frame, then 10 seconds later (at t=12) the actual movie began. So this would correspond to a 
dumm_offset of -10. Double check this with reliability results.

Due to the late start of the fMRI scanner (2s after the movie already started),
we have no data for clips 0s-2s.

Each segment and repetition is analyzed separately. Ideally I could do GLM on both reps and use
GLMsingle CD, but combining runs across sessions can get noisy.
"""
def vectorized_correlation(x,y,dim=0):
    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, ddof=1, keepdims=True)
    y_std = y.std(axis=dim, ddof=1, keepdims=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr.ravel()

def intersubject_correlation(fmri_data, sub_idx):
    """
    perform pairwise correlations between the subject's repetition
    averaged timeseries with the other subject's repetition averaged 
    timeseries. Average all correlations to get that subject's 
    intersubject correlation.
    Inputs:
    fmri_data: array of shape (n subjects, n timepoints, n vertices)
    sub_idx: the index of fmri_data that correspondds to the subject
        you want to calculate the intersubject correlation for.
    
    Returns:
    isc: array of shape (n vertices,) of the intersubject correlation at each vertex
    """
    nsub = fmri_data.shape[0]
    numvertices = fmri_data.shape[2]
    corr_tmp = np.zeros((nsub-1, numvertices)) #shape (nsub-1, numvertices)
    left_out = fmri_data[sub_idx,:,:]
    remaining = np.delete(fmri_data, sub_idx, axis=0)
    for i in range(nsub-1):
        corr_tmp[i, :] = vectorized_correlation(left_out, remaining[i,:,:])

    return corr_tmp.mean(axis=0)

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

class arguments():
    def __init__(self) -> None:
        self.root = "/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset"
        self.task = "train" #"test" is not yet supported
        self.plot = True
        self.verbose = True
args = arguments()
nvertices = 91282

if args.task == 'train':
    numreps = 2
    nRepeatedClips = 4302

subject_maxvals = {'subject1': 0.86, 'subject2': 0.78, 'subject3': 0.73}

fmri_data = np.zeros((3, nRepeatedClips, nvertices))
for sub_idx, sub in enumerate(range(1,4)):
    subject_isc = f"subject{sub}"
    if args.verbose:
        print(f"loading betas from subject {sub}")
    #load raw ts estimates
    with open(os.path.join(args.root, "TSTrialEstimates", subject_isc, "estimates-prepared", "step01", f"{subject_isc}_z=0_TSTrialEstimates_task-{args.task}.pkl"), 'rb') as f:
        estimates_noz_test, condition_order = pickle.load(f) #shape nvideos, nvertices 

    repeated_conditions = list_rep(condition_order, numreps) #just get the conditions repeated exactly numpreps times
    estimates_noz_test_matrix = np.zeros((nvertices, numreps, len(repeated_conditions)))
    for count, cond in enumerate(repeated_conditions):
        idx = [i for i, c in enumerate(condition_order) if c == cond] #all indices of repeated condition "cond"
        estimates_noz_test_matrix[:,:,count] = estimates_noz_test[idx, :].T
    fmri_data[sub_idx, :, :] = estimates_noz_test_matrix.mean(axis=1).T #average over the stimuli repetitions

#this is valid for training only because all subjects saw same repeats in training data but not testing data
segments = {f"seg{s}": [] for s in range(1,19)}
for idx, r in enumerate(repeated_conditions):
    seg = r.split('_')[0]
    segments[seg].append(idx)

for sub_idx, sub in enumerate(range(1,4)):
    subject_isc = f"subject{sub}"
    if args.verbose:
        print(f"starting ISC computation for {subject_isc}")

    #first only do the correlations within segments
    isc = 0
    for segment, idx in segments.items():
        isc += intersubject_correlation(fmri_data[:,idx,:], sub_idx)
    isc = isc/len(segments) #average over segments

    if args.plot:
        views = ['lateral', 'medial'] #['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior']
        if args.verbose:
            print("plotting ISC results")
        plot_root = os.path.join(args.root, "ISC_TS", subject_isc)
        if not os.path.exists(plot_root):
            os.makedirs(plot_root)
        task = args.task
        isc[isc < 0] = 0 #treshold by 0
        stat = isc
        max_val = subject_maxvals[subject_isc] #np.nanmax(stat)
        #inflated brain
        for hemi in ['left','right']:
            mesh = hcp.mesh.inflated
            cortex_data = hcp.cortex_data(stat)
            bg = hcp.mesh.sulc
            for view in views:
                display = plotting.plot_surf_stat_map(mesh, cortex_data, hemi=hemi,
                threshold=0.01, bg_map=bg, view=view, vmax=max_val)

                colorbar = display.axes[1]
                custom_ticks = np.array([0, np.nanmax(stat), max_val])
                custom_labels = [f"{tick:.2f}" for tick in custom_ticks]
                colorbar.set_yticks(custom_ticks)
                colorbar.set_yticklabels(custom_labels)

                plt.savefig(os.path.join(plot_root, f"{subject_isc}_isc_task-{task}_mesh-inflated_hemi-{hemi}_view-{view}.png"), dpi=300)
                #plt.savefig(os.path.join(plot_root, f"{subject_isc}_isc_task-{task}_mesh-inflated_hemi-{hemi}_view-{view}.svg"))
            #flattened brain
            if hemi == 'left':
                cortex_data = hcp.left_cortex_data(stat)
                display = plotting.plot_surf(hcp.mesh.flat_left, cortex_data,
                threshold=0.01, bg_map=hcp.mesh.sulc_left, colorbar=True, cmap='hot', vmax=max_val)

                colorbar = display.axes[1]
                custom_ticks = np.array([0, np.nanmax(stat), max_val])
                custom_labels = [f"{tick:.2f}" for tick in custom_ticks]
                colorbar.set_yticks(custom_ticks)
                colorbar.set_yticklabels(custom_labels)

                plt.savefig(os.path.join(plot_root, f"{subject_isc}_isc_task-{task}_mesh-flat_hemi-left.png"), dpi=300)
                #plt.savefig(os.path.join(plot_root, f"{subject_isc}_isc_task-{task}_mesh-flat_hemi-left.svg"))
            if hemi == 'right':
                cortex_data = hcp.right_cortex_data(stat)
                display = plotting.plot_surf(hcp.mesh.flat_right, cortex_data,
                threshold=0.01, bg_map=hcp.mesh.sulc_right, colorbar=True, cmap='hot', vmax=max_val)

                colorbar = display.axes[1]
                custom_ticks = np.array([0, np.nanmax(stat), max_val])
                custom_labels = [f"{tick:.2f}" for tick in custom_ticks]
                colorbar.set_yticks(custom_ticks)
                colorbar.set_yticklabels(custom_labels)

                plt.savefig(os.path.join(plot_root, f"{subject_isc}_isc_task-{task}_mesh-flat_hemi-right.png"), dpi=300)
                #plt.savefig(os.path.join(plot_root, f"{subject_isc}_isc_task-{task}_mesh-flat_hemi-right.svg"))