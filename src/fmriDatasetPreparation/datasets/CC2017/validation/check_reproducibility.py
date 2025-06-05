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
import argparse
import nibabel as nib
from scipy.stats import pearsonr
from scipy.stats import zscore
from tqdm import tqdm
import hcp_utils as hcp
from nilearn import plotting

#Goal: get beta estimates for 2s chunks of continuous video
#perform GLM on each segment separately, as provided in the data release, no on the entire run
#as acquired


#compute glm using glmsingle.
# https://github.com/cvnlab/GLMsingle
#helpful nibabel + cifti tutorial: https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
#other cifti links:
# http://www.nitrc.org/projects/cifti/
# https://www.humanconnectome.org/software/workbench-command/-cifti-help

"""
    Parcels: each index refers to a named subset of the brainordinates (i.e.
        'V1', and the surface vertices in V1)
    Scalars: each index is simply given a name (i.e. 'Myelin')
    Series: each index is assigned a quantity in a linear series (i.e., a
        timeseries of 0 sec, 0.7 sec, 1.4 sec, ...)
    Labels: each index is assigned a name (i.e., 'Visual Areas'), but also a
        list of labels that maps integer data values to names and colors (i.e.
        {(5, 'V1', #ff0000), (7, 'V2', #00ff00), ...}

   The common types of cifti files and the mapping types they use are:
      dconn: ROW is dense, COLUMN is dense
      dscalar: ROW is scalars, COLUMN is dense
      dtseries: ROW is series, COLUMN is dense
      dlabel: ROW is labels, COLUMN is dense
      pconn: ROW is parcels, COLUMN is parcels
      pdconn: ROW is dense, COLUMN is parcels
      dpconn: ROW is parcels, COLUMN is dense
      pscalar: ROW is scalars, COLUMN is parcels
      ptseries: ROW is series, COLUMN is parcels

"""

"""
load data
cifti = nb.load(fname)
cifti_data = cifti.get_fdata(dtype=np.float32)
cifti_hdr = cifti.header
nifti_hdr = cifti.nifti_header

save the data
img = cifti2.Cifti2Image(cifti_data, header=cifti_hdr)
img.to_filename(’/path/test.dtseries.nii’)

"""

#arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-10 that you wish to process")
parser.add_argument("-c", "--root", default="/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset", help="The root path to the cifti files")
parser.add_argument("-t", "--task", default='train', required=True, help="test or train task that you are processing")
parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
parser.add_argument("-p", "--plot", action="store_true", help="Bool to plot noise ceiling")

args = parser.parse_args()
"""
class arguments():
    def __init__(self) -> None:
        self.subject = "1"
        self.root = "/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset"
        self.task = "train"
        self.verbose = True
        self.plot = True
args = arguments()
"""
subject = f"subject{args.subject}"
if args.verbose:
    print(f"Running single trial estimates on CC2017 Wen dataset main task for subject {subject}")
views = ['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior']

TR = 2 #acquisition TR
TR_resamp = 2 # resample time series to be locked to stimulus onset
videoChunk = 2 #what snippet length you want to chunk the continuous video into
stim_length = 480 #in seconds, how long is the stimuli segment. does not account for when fmri acquisition started, just the length of the mp4 file
stimDur = 2 #in seconds. Parameter for GLM. could be impulse (0)
dummy_offset = 2 #in seconds. for most runs, the fmri acquisision started 2s after the movie started
BOLDoffset = 4 #in seconds, what is the offset of the peak BOLD signal
nvertices = 91282

sub_save_root = os.path.join(args.root, "TSTrialEstimates", subject)
if not os.path.exists(sub_save_root):
    os.makedirs(sub_save_root)

if args.task == 'train':
    segments = [f"seg{rep}" for rep in range(1,19)]
    numrepeats = 2
elif args.task == 'test':
    segments = [f"test{rep}" for rep in range(1,6)]
    numrepeats = 10

segs = []
for count, segment in enumerate(segments):
    reps = []
    for rep in range(1,numrepeats+1):
        stim_order = []
        rep_corr = np.zeros((nvertices,))
        print(f"{subject} segment {segment} repetition {rep}")
        #overwrite defaults if special task, run, or rep. See the README Take Caution section
        if segment in ['test2', 'test3', 'test4', 'test5']:
            dummy_offset=0 #this overwrites the above dummy_offset default

        #load data
        cifti = nib.load(os.path.join(args.root, subject, "fmri", segment, "cifti", f"{segment}_{rep}_Atlas.dtseries.nii"))
        reps.append(cifti.get_fdata(dtype=np.float32))
    
    for v in tqdm(range(nvertices)):
        rep_corr[v] = pearsonr(reps[0][:,v], reps[1][:,v])[0]
    del reps
    segs.append(rep_corr)
    #rep_corr_z = zscore(rep_corr, axis=0, ddof=1, nan_policy='propagate')
    #segs.append(rep_corr_z)

reproducibility = np.mean(np.array(segs), axis=0)

if args.plot:
    if args.verbose:
        print("plotting noiseceling results")
    plot_root = os.path.join(sub_save_root, "plots_reproducibility")
    if not os.path.exists(plot_root):
        os.makedirs(plot_root)
    task = args.task
    stat = reproducibility.copy()
    for hemi in ['left','right']:
        mesh = hcp.mesh.inflated
        cortex_data = hcp.cortex_data(stat)
        bg = hcp.mesh.sulc
        for view in views:
            plotting.plot_surf_stat_map(mesh, cortex_data, hemi=hemi,
            threshold=0.015, bg_map=bg, view=view,
            output_file=os.path.join(plot_root, f"{subject}_reproducibility_task-{task}_hemi-{hemi}_view-{view}.png"))

