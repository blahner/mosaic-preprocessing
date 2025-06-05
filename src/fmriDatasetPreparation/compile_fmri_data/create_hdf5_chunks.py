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

    no_noiseceilings = [f"sub-{s:02}_HAD" for s in range(1,31)] + [f"sub-{s:02}_NOD" for s in range(10,31)] #subjects where no noise ceilings exist (no stim repeats)
    filenames = glob.glob(os.path.join(args.root, "betas_fsLR32k", "*.npz")) #note that order is not preserved
    groups_unique = list(set(['_'.join(Path(fname).name.split('_')[:2]) for fname in filenames]))
    groups = sorted(groups_unique, key=lambda x: (x.split('_')[1], int(x.split('_')[0].replace('sub-', ''))))
    assert len(groups) == 93, f"Expected 93 groups but found {len(groups)}"
    # Create a file
    print("starting hdf5 file creation...")
    with h5py.File(os.path.join(args.root,'mosaic_version-1_0_0_chunks_new.hdf5'), 'w') as f:
        for group in tqdm(groups, total=len(groups), desc="Adding 'sub-XX_DATASET' groups to hdf5 file"):
            if group in f:
                continue
                
            print(f"Working on group: {group}")
            subject, fmri_dataset_name = group.split('_')
            participant_info = pd.read_table(os.path.join(args.root, "participants", short_to_long[fmri_dataset_name], "participants.tsv"))
            grp = f.create_group(group, track_order=True)
            grp.attrs.create("visual_angle", visual_angles[fmri_dataset_name])
            grp.attrs.create('sex', participant_info.loc[participant_info['participant_id']==subject,'sex'].values[0])
            grp.attrs.create('age', participant_info.loc[participant_info['participant_id']==subject,'age'].values[0])

            group_response_filenames = sorted(glob.glob(os.path.join(args.root, "betas_fsLR32k", f"{group}*_nanfilled-*.npz"))) #note that order is not preserved
            assert len(group_response_filenames) > 0, f"Expected more than 0 files, found {len(group_response_filenames)}"
            
            group_noiseceiling_filenames = sorted(glob.glob(os.path.join(args.root, "noiseceilings", f"{group}*_nanfilled-*.npz"))) #note that order is not preserved
            if group not in no_noiseceilings:
                assert len(group_noiseceiling_filenames) > 0, f"Expected more than 0 noiseceiling files, found {len(group_noiseceiling_filenames)}"
                grp_noiseceiling = grp.create_group(f"noiseceilings", track_order=True)

                for noiseceiling_filename in group_noiseceiling_filenames:
                    #all noiseceilings are either '1' or 'avg'
                    core_filename = Path(noiseceiling_filename).stem
                    if '_n-1_' in core_filename:
                        n = '1'
                    elif '_n-avg_' in core_filename:
                        n = 'avg'
                    else:
                        raise ValueError(f"n should be 1 or avg, but neither option was found in file {core_filename}.")
                    phase_n_noiseceiling = core_filename.split(f"{group}_phase-")
                    phase = phase_n_noiseceiling[-1].split(f'_n-{n}')[0]

                    data = np.load(noiseceiling_filename) #has keys 'response' and 'nan_indices'
                    dset = grp_noiseceiling.create_dataset(core_filename.split('_nanfilled-')[0], data=data['response'], track_order=True)
                    dset.attrs.create('nan_indices', data['nan_indices'])
                    dset.attrs.create('n', n)
                    dset.attrs.create('phase', phase)
            
            grp_nan_indices = grp.create_group("nan_indices", track_order=True)

            betas_chunk = np.zeros((len(group_response_filenames), 91282))
            presented_stimulus_chunk = []
            image_stimulus_chunk = []
            for idx, response_filename in enumerate(group_response_filenames):
                #"sub-01_BMD_stimulus-0-2-4-4-9-1-5-4-6102449154_phase-test_rep-0.npz"
                core_filename = Path(response_filename).stem
                stim_phase_rep = core_filename.split(f'{group}_stimulus-')[-1]
                phase_rep = stim_phase_rep.split('_phase-')
                stimulus_name = phase_rep[0]
                phase, rep_extra = phase_rep[1].split('_rep-')
                rep = rep_extra.split('_nanfilled-')[0]
                if fmri_dataset_name in ['BMD','HAD']:
                    image_stimulus_filename_tmp = glob.glob(os.path.join(args.root,"stimuli", "frames_middle", f"{stimulus_name}*.jpg"))
                    assert(len(image_stimulus_filename_tmp) == 1)
                    image_stimulus_filename = Path(image_stimulus_filename_tmp[0]).name
                    presented_stimulus_filename_tmp = glob.glob(os.path.join(args.root,"stimuli", "raw", f"{stimulus_name}.*"))
                    assert(len(presented_stimulus_filename_tmp) == 1)
                    presented_stimulus_filename = Path(presented_stimulus_filename_tmp[0]).name
                else:
                    image_stimulus_filename_tmp = glob.glob(os.path.join(args.root, "stimuli", "raw", f"{stimulus_name}.*"))
                    assert(len(image_stimulus_filename_tmp) == 1)
                    image_stimulus_filename = Path(image_stimulus_filename_tmp[0]).name
                    presented_stimulus_filename = Path(image_stimulus_filename_tmp[0]).name
                data = np.load(response_filename) #has keys 'response' and 'nan_indices'
                betas_chunk[idx,:] = data['response']

                presented_stimulus_chunk.append(presented_stimulus_filename)
                image_stimulus_chunk.append(image_stimulus_filename)
                grp_nan_indices.create_dataset(core_filename.split('_nanfilled-')[0], data=data['nan_indices'], track_order=True)

            # Brain responses dataset
            grp.create_dataset(
                'betas',
                data=betas_chunk,
                dtype='float32',
                chunks=(100, 91282),
                track_order=True
            )
            
            # Stimulus names as variable-length strings
            grp.create_dataset(
                'image_stimulus_filename',
                data=np.array(image_stimulus_chunk, dtype=h5py.string_dtype()),
                dtype=h5py.string_dtype(),
                chunks=(100,),
                track_order=True
            )
            grp.create_dataset(
                'presented_stimulus_filename',
                data=np.array(presented_stimulus_chunk, dtype=h5py.string_dtype()),
                dtype=h5py.string_dtype(),
                chunks=(100,),
                track_order=True 
            )
 

if __name__=='__main__':
    datasets_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC") #use default if DATASETS_ROOT env variable is not set.

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=datasets_root_default, help="Root path to scratch datasets folder.")

    args = parser.parse_args()
    
    main(args)