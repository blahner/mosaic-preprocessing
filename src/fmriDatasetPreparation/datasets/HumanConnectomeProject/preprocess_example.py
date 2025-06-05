import nibabel as nib
import numpy as np
import hcp_utils as hcp
import os
from tqdm import tqdm

# HCP_DATA_DIR = '/data/vision/oliva/scratch/datasets/hcp_dataset/HCP_1200/'
HCP_DATA_DIR = "./hp2000_clean/"
ROOT_DIR = './output_npz/'
IMG_PATH = 'MNINonLinear/Results/rfMRI_%s/rfMRI_%s_Atlas_MSMAll_hp2000_clean.dtseries.nii'

v1_idx = np.where(((hcp.mmp.map_all == 1)) | (hcp.mmp.map_all == 181))[0] 
v2_idx = np.where(((hcp.mmp.map_all == 4)) | (hcp.mmp.map_all == 184))[0]
v3_idx = np.where(((hcp.mmp.map_all == 5)) | (hcp.mmp.map_all == 185))[0]
v4_idx = np.where(((hcp.mmp.map_all == 6)) | (hcp.mmp.map_all == 186))[0]

sub_list = os.listdir(HCP_DATA_DIR)
num_npz = 0
for sub in tqdm(sub_list):
    for area_name in ['REST1_LR']:
        img_path = os.path.join(HCP_DATA_DIR, sub, IMG_PATH % (area_name, area_name))
        
        if not os.path.exists(img_path):
            continue

        img = nib.load(img_path)
        X = img.get_fdata()
        X = hcp.normalize(X)
        output_dir = os.path.join(ROOT_DIR, 'npz', sub, area_name)
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            os.path.join(output_dir, 'HCP_visual_voxel.npz'),
            V1=X[:,v1_idx],
            V2=X[:,v2_idx],
            V3=X[:,v3_idx],
            V4=X[:,v4_idx]
        )
        num_npz += 1

    print("Number of example: {}".format(num_npz))