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

class arguments():
    def __init__(self) -> None:
        self.root = "/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset"
        self.task = "train"
        self.plot = True
        self.verbose = True
args = arguments()

for sub in range(1,4):
    subject = f"subject{sub}"

    if args.task == 'train':
        numsegments = 18
        segments = [f"seg{rep}" for rep in range(1,numsegments+1)]
        numrepeats = 2

    intrasbject_corr = 0
    if args.verbose:
        print(f"loading data from {subject}")
    for count, segment in enumerate(segments):
        fmri_data_seg = 0
        if args.verbose:
            print(f"loading segment {segment}")
        
        fmri_rep1 = nib.load(os.path.join(args.root, subject, "fmri", segment, "cifti", f"{segment}_1_Atlas.dtseries.nii")).get_fdata(dtype=np.float32)
        fmri_rep2 = nib.load(os.path.join(args.root, subject, "fmri", segment, "cifti", f"{segment}_2_Atlas.dtseries.nii")).get_fdata(dtype=np.float32)

        intrasbject_corr += vectorized_correlation(fmri_rep1, fmri_rep2) #accumulate the correlations between reps over all segments

    intrasbject_corr = intrasbject_corr/numsegments #average over the segments

    if args.plot:
        views = ['lateral', 'medial'] #['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior']
        if args.verbose:
            print("plotting Intrasubject Correlation results")
        plot_root = os.path.join(args.root, "IntraSC", subject)
        if not os.path.exists(plot_root):
            os.makedirs(plot_root)
        task = args.task
        intrasbject_corr[intrasbject_corr < 0] = 0 #treshold by 0
        stat = intrasbject_corr
        max_val = np.nanmax(stat)
        #inflated brain
        for hemi in ['left','right']:
            mesh = hcp.mesh.inflated
            cortex_data = hcp.cortex_data(stat)
            bg = hcp.mesh.sulc
            for view in views:
                display = plotting.plot_surf_stat_map(mesh, cortex_data, hemi=hemi,
                threshold=0.01, bg_map=bg, view=view, vmax=max_val)
                plt.savefig(os.path.join(plot_root, f"{subject}_itrasc_task-{task}_mesh-inflated_hemi-{hemi}_view-{view}.png"), dpi=300)
                #plt.savefig(os.path.join(plot_root, f"{subject}_intrasc_task-{task}_mesh-inflated_hemi-{hemi}_view-{view}.svg"))
            #flattened brain
            if hemi == 'left':
                cortex_data = hcp.left_cortex_data(stat)
                display = plotting.plot_surf(hcp.mesh.flat_left, cortex_data,
                threshold=0.01, bg_map=hcp.mesh.sulc_left, colorbar=True, cmap='hot', vmax=max_val)
                plt.savefig(os.path.join(plot_root, f"{subject}_intrasc_task-{task}_mesh-flat_hemi-left.png"), dpi=300)
                #plt.savefig(os.path.join(plot_root, f"{subject}_intrasc_task-{task}_mesh-flat_hemi-left.svg"))
            if hemi == 'right':
                cortex_data = hcp.right_cortex_data(stat)
                display = plotting.plot_surf(hcp.mesh.flat_right, cortex_data,
                threshold=0.01, bg_map=hcp.mesh.sulc_right, colorbar=True, cmap='hot', vmax=max_val)
                plt.savefig(os.path.join(plot_root, f"{subject}_intrasc_task-{task}_mesh-flat_hemi-right.png"), dpi=300)
                #plt.savefig(os.path.join(plot_root, f"{subject}_intrasc_task-{task}_mesh-flat_hemi-right.svg"))