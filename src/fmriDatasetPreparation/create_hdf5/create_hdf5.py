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
import warnings

"""
Generate a subject's hdf5 file that contains beta estimates, noise ceilings, and all the required attributes.
"""

def main(args):
    # Create a file
    print(f"starting hdf5 file creation for {args.subjectID_dataset}...")
    with h5py.File(os.path.join(args.root, 'hdf5_files', f'{args.subjectID_dataset}.hdf5'), 'w') as grp:
        subjectID, fmri_dataset_name = args.subjectID_dataset.split('_')
        #First add attributes
        grp.attrs.create("visual_angle", args.visual_angle)
        grp.attrs.create('sex', args.sex)
        grp.attrs.create('age', args.age)
        grp.attrs.create('publication_url', args.publication_url)
        grp.attrs.create('github_url', args.github_url)
        grp.attrs.create('owner_name', args.owner_name)
        grp.attrs.create('owner_email', args.owner_email)
        grp.attrs.create('dataset_name', args.dataset_name)
        grp.attrs.create('subjectID', subjectID)
        grp.attrs.create('preprocessing_pipeline', args.preprocessing_pipeline)
        grp.attrs.create('beta_pipeline', args.beta_pipeline)
        
        #do beta assertion early
        group_response_filenames = sorted(glob.glob(os.path.join(args.root, "betas_fsLR32k", f"{args.subjectID_dataset}*_nanfilled-*.npz"))) #note that order is not preserved
        assert len(group_response_filenames) > 0, f"Expected more than 0 files"

        #second add noisceilings, if applicable
        if args.subjectID_dataset not in args.no_noiseceilings:
            group_noiseceiling_filenames = sorted(glob.glob(os.path.join(args.root, "noiseceilings", f"{args.subjectID_dataset}*_nanfilled-*.npz"))) #note that order is not preserved
            assert len(group_noiseceiling_filenames) > 0, f"Expected more than 0 noiseceiling files, found {len(group_noiseceiling_filenames)}"
            grp_noiseceiling = grp.create_group(f"noiseceilings", track_order=True)
        
            for noiseceiling_filename in tqdm(group_noiseceiling_filenames, total=len(group_noiseceiling_filenames), desc="Looping over noise ceiling files"):
                #all noiseceilings are either '1' or 'avg', representing the number of trials that are averaged over for the noiseceiling calculation
                core_filename = Path(noiseceiling_filename).stem
                if '_n-1_' in core_filename:
                    n = '1'
                elif '_n-avg_' in core_filename:
                    n = 'avg'
                else:
                    raise ValueError(f"n should be 1 or avg, but neither option was found in file {core_filename}.")
                phase_n_noiseceiling = core_filename.split(f"{args.subjectID_dataset}_phase-")
                phase = phase_n_noiseceiling[-1].split(f'_n-{n}')[0]

                #load the noise ceiling data
                data = np.load(noiseceiling_filename) #has keys 'response' and 'nan_indices'
                
                #add to hdf5 file
                dset = grp_noiseceiling.create_dataset(core_filename.split('_nanfilled-')[0], data=data['response'], track_order=True)
                dset.attrs.create('nan_indices', data['nan_indices'])
                dset.attrs.create('n', n)
                dset.attrs.create('phase', phase)
        
        #third, add single trial beta estimates
        grp_betas = grp.create_group(f"betas", track_order=True)
        for response_filename in tqdm(group_response_filenames, total=len(group_response_filenames), desc="Looping over beta files"):
            #example response filename: "sub-01_BMD_stimulus-0-2-4-4-9-1-5-4-6102449154_phase-test_rep-0_nanfilled-adjacency.npz"
            core_filename = Path(response_filename).stem
            stim_phase_rep = core_filename.split(f'{args.subjectID_dataset}_stimulus-')[-1]
            phase_rep = stim_phase_rep.split('_phase-')
            stimulus_name = phase_rep[0]
            phase, rep_extra = phase_rep[1].split('_rep-')
            rep = rep_extra.split('_nanfilled-')[0]

            #confirm we are only getting stimulus filename
            presented_stimulus_filename_tmp = glob.glob(os.path.join(args.root, "stimuli", "raw", f"{stimulus_name}.*"))
            assert(len(presented_stimulus_filename_tmp) == 1)

            #load the beta response
            data = np.load(response_filename) # has keys 'response' and 'nan_indices'

            #add to hdf5 file
            dset = grp_betas.create_dataset(core_filename.split('_nanfilled-')[0], data=data['response'], track_order=True)
            dset.attrs.create('nan_indices', data['nan_indices'])
            dset.attrs.create('phase', phase)
            dset.attrs.create('repetition', rep)
            dset.attrs.create('presented_stimulus_filename', Path(presented_stimulus_filename_tmp[0]).name)

if __name__=='__main__':
    root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC") #use default if DATASETS_ROOT env variable is not set.

    parser = argparse.ArgumentParser()
    parser.add_argument("--subjectID_dataset", type=str, required=True, help="Name of dataset you want to create the hdf5 for in format sub-XX_DATASET. For example, sub-01_NSD.")
    parser.add_argument("--owner_name", type=str, required=True, default=None, help="Name of the creator of this .hdf5 file. Does not need to be an author of the original publication.")
    parser.add_argument("--owner_email", type=str, required=True, default=None, help="Email of the owner of this .hdf5 file where users can contact you.")
    parser.add_argument("--age", type=str, default=None, help="Age of participant at time of data collection (years).")
    parser.add_argument("--sex", type=str, default=None, help="Reported sex of participant.")
    parser.add_argument("--visual_angle", type=str, default=None, help="Visual angle the stimuli were presented to the participant.")
    parser.add_argument("--publication_url", type=str, default=None, help="URL to the original publication of the fMRI dataset.")
    parser.add_argument("--github_url", type=str, default=None, help="URL to the public GitHub repository containing the code used to process the fMRI data contained in this .hdf5 file..")
    parser.add_argument("--preprocessing_pipeline", type=str, default=None, help="Pipeline used to preprocess the fMRI data contained in this .hdf5 file, e.g., 'fMRIPrepv23.2.0'")
    parser.add_argument("--beta_pipeline", type=str, default=None, help="Pipeline used to estimate the beta values contained in this .hdf5 file, e.g., 'GLMsinglev1.2'")
    parser.add_argument("--root", type=str, default=root_default, help="Root path to datasets folder.")

    args = parser.parse_args()

    subjectID, dataset = args.subjectID_dataset.split('_') # 'dataset' should be the short name of the dataset.
    subnum = int(subjectID.split('-')[-1])

    #it might be easier to just fill out a dictionary here if you are repeatedly running this script
    short_to_long = {"BOLD5000": "BOLD5000", "BMD": "BOLDMomentsDataset", "NSD": "NaturalScenesDataset", "GOD": "GenericObjectDecoding",
                     "NOD": "NaturalObjectDataset", "HAD": "HumanActionsDataset", "THINGS": "THINGS_fmri", "deeprecon": "deeprecon"}
    nsubjects = {"BOLD5000": 4, "BMD": 10, "NSD": 8, "GOD": 5, "NOD": 30, "HAD": 30, "THINGS": 3, "deeprecon": 3}
    publication_urls = {"BOLD5000": "https://www.nature.com/articles/s41597-019-0052-3",
                        "BMD": "https://www.nature.com/articles/s41467-024-50310-3",
                        "NSD": "https://www.nature.com/articles/s41593-021-00962-x",
                        "GOD": "https://www.nature.com/articles/ncomms15037",
                        "NOD": "https://www.nature.com/articles/s41597-023-02471-x",
                        "HAD": "https://www.nature.com/articles/s41597-023-02325-6",
                        "THINGS": "https://elifesciences.org/articles/82580",
                        "deeprecon": "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633"}
    
    github_urls = {"BOLD5000": "https://github.com/blahner/mosaic-preprocessing",
                        "BMD": "https://github.com/blahner/mosaic-preprocessing",
                        "NSD": "https://github.com/blahner/mosaic-preprocessing",
                        "GOD": "https://github.com/blahner/mosaic-preprocessing",
                        "NOD": "https://github.com/blahner/mosaic-preprocessing",
                        "HAD": "https://github.com/blahner/mosaic-preprocessing",
                        "THINGS": "https://github.com/blahner/mosaic-preprocessing",
                        "deeprecon": "https://github.com/blahner/mosaic-preprocessing"}

    pipelines = {"https://github.com/blahner/mosaic-preprocessing": {"preprocessing_pipeline": "fMRIPrepv23.2.0", "beta_pipeline": "GLMsinglev1.2"}}
    
    #check if valid subject and dataset was entered
    if dataset not in short_to_long:
        raise ValueError(f"dataset {dataset} not recognized. Must be one of {list(short_to_long.keys())} or add it here.")
    if subnum > nsubjects[dataset] or subnum < 1:
        raise ValueError(f"Invalid subjectID. Dataset {dataset} is defined for subjects 01 through {nsubjects[dataset]:02} inclusive. You attempted to process subject {subjectID}")
    
    args.dataset_name = short_to_long[dataset] #add the dataset's full name

    if not args.visual_angle:
        visual_angles = {"BOLD5000": 4.6, "BMD": 5, "NSD": 8.4, "GOD": 12, "NOD": 16, "HAD": 16, "THINGS": 10, "deeprecon": 12}
        args.visual_angle = visual_angles[dataset]
    
    args.no_noiseceilings = [f"sub-{s:02}_HAD" for s in range(1,31)] + [f"sub-{s:02}_NOD" for s in range(10,31)] #subjects where no noise ceilings exist (no stim repeats)

    if not args.sex or args.age:
        #use the fMRI dataset's participants.tsv file, required for BIDS format, to get subject age and sex information. Otherwise, most manuscripts say this in methods.
        participant_info = pd.read_table(os.path.join(args.root, "participants", short_to_long[dataset], "participants.tsv"))
        if not args.sex:
            args.sex = participant_info.loc[participant_info['participant_id']==subjectID,'sex'].values[0]
        if not args.age:
            args.age = participant_info.loc[participant_info['participant_id']==subjectID,'age'].values[0]

    if not args.publication_url:
        args.publication_url = publication_urls[dataset]
    
    if not args.github_url:
        args.github_url = github_urls[dataset]

    if not args.preprocessing_pipeline:
        args.preprocessing_pipeline = pipelines[args.github_url]['preprocessing_pipeline']

    if not args.beta_pipeline:
        args.beta_pipeline = pipelines[args.github_url]['beta_pipeline']
    
    main(args)