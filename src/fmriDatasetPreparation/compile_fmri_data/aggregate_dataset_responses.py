from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import argparse
import pickle
from pathlib import Path

"""
Copy single trial brain responses to central dataset location. Each trial
is test or train based on how it was normalized after the GLM.
"""

def main(args):
    target_root = os.path.join(args.datasets_root, "MOSAIC")
    save_path = os.path.join(target_root, "betas_fsLR32k")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fmri_datasets = {"BOLD5000": [4, "BOLD5000"], 
                     "NaturalObjectDataset": [30, "NOD"], 
                     "HumanActionsDataset": [30, "HAD"], 
                     "BOLDMomentsDataset": [10, "BMD"], 
                     "NaturalScenesDataset": [8, "NSD"], 
                     "GenericObjectDecoding": [5, "GOD"], 
                     "deeprecon": [3, "deeprecon"], 
                     "THINGS_fmri": [3, "THINGS"]} 

    for dataset_longname, value in fmri_datasets.items():
        numsubjects, dataset_shortname = value
        print(f"starting {dataset_shortname}")
        for sub in range(1,numsubjects+1):
            if dataset_shortname == "BOLD5000":
                subject = f"sub-CSI{sub}"
            else:
                subject = f"sub-{sub:02}"
            print(f"running subject {subject}")
            for task in ['test', 'train', 'artificial']:
                #load testing/training responses
                if dataset_shortname == "BMD":
                    fmri_root = os.path.join(args.datasets_root, dataset_longname, "derivatives", "versionC", "GLM", subject, "prepared_betas")
                else:
                    fmri_root = os.path.join(args.datasets_root, dataset_longname, "derivatives", "GLM", subject, "prepared_betas")
                with open(os.path.join(fmri_root, f"{subject}_organized_betas_task-{task}_normalized.pkl"), 'rb') as f:
                    betas, stimorder = pickle.load(f)
                numstim, numreps, numvertices = betas.shape
                assert(numstim == len(stimorder))
                if task == 'test' or task == 'artificial':
                    phase = 'test'
                elif task == 'train':
                    phase = 'train'
                for stim in range(numstim):
                    for rep in range(numreps):
                        response = betas[stim, rep, :]
                        if np.isnan(response).all(): #all nan responses were filler responses just to keep the matrix dimensionality
                            continue
                        stim_name = Path(stimorder[stim]).stem
                        np.save(os.path.join(save_path, f"sub-{sub:02}_{dataset_shortname}_stimulus-{stim_name}_phase-{phase}_rep-{rep}.npy"), response)

if __name__=='__main__':
    datasets_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets")) #use default if DATASETS_ROOT env variable is not set.

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", type=str, default=datasets_root_default, help="Root path to scratch datasets folder.")

    args = parser.parse_args()
    
    main(args)