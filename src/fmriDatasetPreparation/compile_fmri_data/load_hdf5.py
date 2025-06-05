from dotenv import load_dotenv
load_dotenv()
import os
import argparse
import h5py
import numpy as np
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm

"""
Save fmri responses in hdf5 format.
"Groups" are defined by the subject and dataset (e.g., "sub-XX_DATASET")
"Datasets" are the individual arrays corresponding to one brain response.
This distinction between groups and datasets follows the naming convention of hdf5 files.
This hdf5 is agnostic to test/train sets or otherwise excluded stimuli.
"""

def main(args):
    short_to_long = {"BOLD5000": "BOLD5000", "BMD": "BOLDMomentsDataset", "NSD": "NaturalScenesDataset", "GOD": "GenericObjectDecoding",
                     "NOD": "NaturalObjectDataset", "HAD": "HumanActionsDataset", "THINGS": "THINGS_fmri", "deeprecon": "deeprecon"}
    visual_angles = {"BOLD5000": 4.6, "BMD": 5, "NSD": 8.4, "GOD": 12, "NOD": 16, "HAD": 16, "THINGS": 10, "deeprecon": 12}
    #assert len(groups) == 93, f"Expected 93 groups but found {len(groups)}"
    # Create a file
    print("loading hdf5 file...")
    with h5py.File(os.path.join(args.root,'mosaic_version-1_0_0_chunks.hdf5'), 'r') as f:
        print(f"Keys: {f.keys()}")
        data = []
        for fname in f['sub-01_GOD'].keys():
            data.append(f['sub-01_GOD'][fname][:10000])

if __name__=='__main__':
    datasets_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC") #use default if DATASETS_ROOT env variable is not set.

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=datasets_root_default, help="Root path to scratch datasets folder.")

    args = parser.parse_args()
    
    main(args)