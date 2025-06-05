import numpy as np
import hcp_utils as hcp
import os
import pandas as pd
import pickle

class arguments():
    def __init__(self) -> None:
        self.beta_type = "typed"
        self.root_dir = '/mnt/t/HAD'
        self.verbose = True
args = arguments()

assert(args.beta_type in ['typea', 'typeb', 'typec', 'typed'])
numstim = 720 #720 unique stimuli presentations per subject in their one session
beta_types = {"typea": "TYPEA_ONOFF", "typeb": "TYPEB_FITHRF" , "typec": "TYPEC_FITHRF_GLMDENOISE", "typed": "TYPED_FITHRF_GLMDENOISE_RR"}
beta_type = beta_types[args.beta_type]

groups = [1,2,3,4,5,6,7,8,9,15,16,17,18] #groups from the glasser parcellation I want. see table 1 in glasser supplementary for details
tmp = pd.read_table(os.path.join("/mnt/t/BMDGeneration/analysis/mindvis/utils","hcp_glasser_roilist.txt"), sep=',')
roi_idx_running = {}
for count, li in enumerate(range(tmp.shape[0])):
    line = tmp.iloc[count,:]
    ROI = line['ROI']
    GROUP = line['GROUP']
    ID = line['ID']
    if GROUP in groups: #if the roi is in a group we want, include that roi
        roi_idx_running[ROI] = np.where(((hcp.mmp.map_all == ID)) | (hcp.mmp.map_all == ID+180))[0]

for sub in range(1,31):
    subject = f"sub-{sub:02}"
    if args.verbose:
        print(f"Preparing MBM parcellation for subject {subject}")
    #load stimulus order
    stimorder_path = os.path.join(args.root_dir, "derivatives", "GLM", "cifti", subject, "GLMsingle", f"{subject}_task-action_conditionOrderDM.npy")
    dm = np.load(stimorder_path, allow_pickle=True)
    stim_order = []
    for run in dm:
        for stim_fname in run['trial_type']:
            stim_order.append(stim_fname)
    assert(len(set(stim_order)) == len(stim_order)) #assert no repeated stimuli filenames
    assert(len(stim_order) == numstim) #each HAD session has 720 videos

    img_path = os.path.join(args.root_dir, "derivatives", "GLM", "cifti", subject, "GLMsingle", f"{beta_type}.npy")
    X = np.load(img_path, allow_pickle=True).item()
    betas = X['betasmd'].T #transpose to put it in shape (#videos, #grayordinates).
    betas_norm = hcp.normalize(betas) #compared to the resting state data, here we treat the stimuli conditions as "time" and normalize so the beta response profile for each grayordinate has mean 0 std 1
    output_dir = os.path.join(args.root_dir, "derivatives","GLM", "cifti", "MBM_preprocess", subject)
    os.makedirs(output_dir, exist_ok=True)
    data = {}
    for roi in roi_idx_running.keys():
        roi_idx = roi_idx_running[roi]
        roi_betas = betas_norm[:, roi_idx]
        assert(np.isnan(roi_betas).any() == False) #dont want any nans in the time series
        data[roi] = roi_betas
    data['stim_order'] = stim_order
    with open(os.path.join(output_dir, 'HAD_visual_voxel_task-action_glmsingle.pkl'),'wb') as f:
        pickle.dump(data, f)