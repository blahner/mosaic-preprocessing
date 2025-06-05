from dotenv import load_dotenv
load_dotenv()
import os
from scipy.sparse import csr_matrix
import argparse
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
import hcp_utils as hcp

#local
from src.utils.transforms import SelectROIs
from helper_functions_nans import fillnans

def main(args):
    roi_indices = SelectROIs(selected_rois=[f"GlasserGroup_{x}" for x in range(1,23)], remove_nan_vertices=False).selected_roi_indices
    assert len(roi_indices) == 59412, f"roi_indices must represent all 59412 vertices of the left and right hemispheres. Your specified {len(roi_indices)}" #confirm we are using all cortical vertices
    
    adjacency_matrix = hcp.cortical_adjacency #get the fsLR32k cortical adjacency matrix
    #cache adjacency as sparse format to improve efficiency
    adjacency_sparse = csr_matrix(adjacency_matrix)

    #precompute adjacency lists for all `roi_indices`
    roi_adjacency = {idx: adjacency_sparse[idx].indices for idx in roi_indices}

    save_root = os.path.join(args.root, "noiseceilings")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    #get all responses that are not nan-filled
    responses_all = glob.glob(os.path.join(args.root, "noiseceilings", f"*_{args.dataset}_*.npy"))
    for response in tqdm(responses_all, total=len(responses_all), desc=f"filling NaNs for {args.dataset} dataset..."):
        filename = Path(response).stem
        output_filename = f"{filename}_nanfilled-{args.fillvalue}.npz"
        #if os.path.isfile(os.path.join(save_root, output_filename)):
        #    continue
        sample = np.load(response)
        fmri_filled, nan_indices = fillnans(sample, roi_indices, roi_adjacency, fillvalue=args.fillvalue)
        #save nan-filled response and nan indices in case user wants to revert
        np.savez(os.path.join(save_root,  output_filename), response=fmri_filled, nan_indices=nan_indices)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="The dataset shortname you want to process, e.g., BMD or NSD.")
    parser.add_argument("-f", "--fillvalue", type=str, required=True, default='adjacency', help="The method you want to use to fill the nans. Must be one of [adjacency, normal_fmri, normal_zero] or a float/int")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)