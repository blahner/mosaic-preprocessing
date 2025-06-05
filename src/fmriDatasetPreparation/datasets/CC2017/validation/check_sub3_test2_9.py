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
#the "minusshift" seems to be correct, where the first 5 scans of subject3 test2_9 
#correspond to 5 scans (10seconds) of still image before the stimuli occurs
#compute LOO
def LOO(data):
    #data is 2d, nsamples x nfeatures
    num_samples = data.shape[0]
    results = np.zeros((num_samples,))
    for samp in range(num_samples):
        left_out = data[samp, :]
        group = np.delete(data, samp, axis=0)
        group_avg = np.mean(group,axis=0)
        results[samp] = pearsonr(left_out, group_avg)[0]
    return results

def LOO9(data):
    #specifically tests rep 9
    num_samples = data.shape[0]
    results = np.zeros((num_samples,))
    rep9 = data[8,:]
    for samp in range(num_samples):
        ind = data[samp,:]
        results[samp] = pearsonr(rep9, ind)[0]
    return results

class arguments():
    def __init__(self) -> None:
        self.subject = "3"
        self.root = "/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset"
        self.task = "test"
        self.verbose = True
args = arguments()

subject = f"subject{args.subject}"
TR = 2 #acquisition TR
TR_resamp = 2 # resample time series to be locked to stimulus onset
videoChunk = 2 #what snippet length you want to chunk the continuous video into
stim_length = 480 #in seconds, how long is the stimuli segment. does not account for when fmri acquisition started, just the length of the mp4 file
stimDur = 2 #in seconds. Parameter for GLM. could be impulse (0)
dummy_offset = 2 #in seconds. for most runs, the fmri acquisision started 2s after the movie started
BOLDoffset = 4 #in seconds, what is the offset of the peak BOLD signal

sub_save_root = os.path.join(args.root, "TSTrialEstimates", subject)
if not os.path.exists(sub_save_root):
    os.makedirs(sub_save_root)

if args.task == 'train':
    segments = [f"seg{rep}" for rep in range(1,19)]
    numrepeats = 2
elif args.task == 'test':
    segments = ["test2"] #[f"test{rep}" for rep in range(1,6)]
    numrepeats = 10

if args.verbose:
    print("defining ROI groups")
groups = list(np.arange(1,23)) #[1,2,3,4,5,6,7,8,9,15,16,17,18] #groups from the glasser parcellation I want. see table 1 in glasser supplementary for details
tmp = pd.read_table(os.path.join(args.root, "utils","hcp_glasser_roilist.txt"), sep=',')
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
with open(os.path.join(args.root, "utils","roi_list_reduced41.txt"), 'r') as f:
    tmp = f.read().splitlines()
for roi in tmp:
    roi = roi[1:-1]
    group_indices.extend(roi_idx_running[roi])
roi_idx_running["Group41"] = np.array(group_indices)

ROI_name = "FFC" #LO1 FFC "V1"

data_minusshift = np.zeros((numrepeats, 241)) #fmri scanner starts BEFORE the stimuli starts. We wait to index the ts starting at scan 5
data_noshift = np.zeros((numrepeats, 241)) #we know this is wrong - there is some isue with rep9. Take the first 241 scans
data_plusshift = np.zeros((numrepeats, 241)) #fmri scanner starts AFTER the stimuli starts. We take the first 241 scans but offset the other reps by 5

for count, segment in enumerate(segments):
    for rep in range(1,numrepeats+1):
        stim_order = []
        print(f"{subject} segment {segment} repetition {rep}")
        #overwrite defaults if special task, run, or rep. See the README Take Caution section
        if segment in ['test2', 'test3', 'test4', 'test5']:
            dummy_offset=0 #this overwrites the above dummy_offset default

        #load data
        cifti = nib.load(os.path.join(args.root, subject, "fmri", segment, "cifti", f"{segment}_{rep}_Atlas.dtseries.nii"))
        cifti_data = cifti.get_fdata(dtype=np.float32)
        
        fmri_interp = cifti_data.copy().T #numvertices x ntime
        if (subject == 'subject3') and (segment == 'test2') and (rep == 9):
            data_minusshift[rep-1,:] = np.mean(fmri_interp[roi_idx_running[ROI_name], 5:],axis=0)
            data_noshift[rep-1,:] = np.mean(fmri_interp[roi_idx_running[ROI_name], :241],axis=0)
            data_plusshift[rep-1,:] = np.mean(fmri_interp[roi_idx_running[ROI_name], :241],axis=0)
            
        else:
            data_minusshift[rep-1,:] = np.mean(fmri_interp[roi_idx_running[ROI_name], :241],axis=0)
            data_noshift[rep-1,:] = np.mean(fmri_interp[roi_idx_running[ROI_name], :241],axis=0)
            data_plusshift[rep-1,:] = np.mean(fmri_interp[roi_idx_running[ROI_name], 5:],axis=0)
print(f"Minus shift:")
print(f"LLO: {np.mean(LOO(data_minusshift),axis=0)}")
print(f"LLO9: {np.mean(LOO9(data_minusshift),axis=0)}")
print(f"LLO9: {LOO9(data_minusshift)}")

print(f"No shift:")
print(f"LLO: {np.mean(LOO(data_noshift),axis=0)}")
print(f"LLO9: {np.mean(LOO9(data_noshift),axis=0)}")
print(f"LLO9: {LOO9(data_noshift)}")

print(f"Plus shift:")
print(f"LLO: {np.mean(LOO(data_plusshift),axis=0)}")
print(f"LLO9: {np.mean(LOO9(data_plusshift),axis=0)}")
print(f"LLO9: {LOO9(data_plusshift)}")
