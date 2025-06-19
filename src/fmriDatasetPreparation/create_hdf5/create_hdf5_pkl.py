from dotenv import load_dotenv
load_dotenv()
import os
import argparse
import h5py
import numpy as np
import glob
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm

"""
Generate a subject's hdf5 file that contains beta estimates, noise ceilings, and all the required attributes from the GLMsingle pickle output.
"""

project_root = os.getenv("PROJECT_ROOT")
def main(args):
    # Create a file
    print(f"starting hdf5 file creation for {args.subjectID_dataset}...")
    hdf5_dir = os.path.join(args.root, "MOSAIC", 'hdf5_files', 'single_subject')
    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)
    with h5py.File(os.path.join(hdf5_dir, f'{args.subjectID_dataset}.hdf5'), 'w') as grp:
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

        nan_indices_all = set() #keep track of all nan indices over all single trial beta estimates and noiseceilings

        #second add noisceilings, if applicable
        if args.subjectID_dataset not in args.no_noiseceilings:
            #load noiseceiling from the dataset validation folder
            noiseceiling_path = os.path.join(project_root, "src", "fmriDatasetPreparation", "datasets", args.dataset_name, "validation", "output", "noiseceiling", subjectID)
            group_noiseceiling_filenames = sorted(glob.glob(os.path.join(noiseceiling_path, "*_n-1_*.npy")) + glob.glob(os.path.join(noiseceiling_path, "*_n-avg_*.npy"))) #note that order is not preserved
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
                task_n_noiseceiling = core_filename.split(f"{args.subjectID_dataset}_phase-")
                task = task_n_noiseceiling[-1].split(f'_n-{n}')[0]

                if task == 'test' or task == 'artificial':
                    phase = 'test'
                elif task == 'train':
                    phase = 'train'

                #load the noise ceiling data
                response = np.load(noiseceiling_filename)
                
                #identify nans
                not_real_mask = np.isnan(response) | np.isinf(response) #True means its not real
                nan_indices = np.where(not_real_mask)[0] #wrt the fsLR32k whole brain space
                nan_indices_all.update(nan_indices)

                #add to hdf5 file
                dset = grp_noiseceiling.create_dataset(core_filename, data=response, track_order=True)
                dset.attrs.create('nan_indices', nan_indices)
                dset.attrs.create('n', n)
                dset.attrs.create('phase', phase)
        
        #third, add single trial beta estimates
        grp_betas = grp.create_group(f"betas", track_order=True)
        if fmri_dataset_name == "BOLD5000":
            subjectID_original = f"sub-CSI{int(subjectID.split('-')[-1])}"
        else:
            subjectID_original = f"sub-{int(subjectID.split('-')[-1]):02}"
        for task in ['test', 'train', 'artificial']:
            #load testing/training responses
            if fmri_dataset_name == "BMD":
                fmri_root = os.path.join(args.root, args.dataset_name, "derivatives", "versionC", "GLM", subjectID_original, "prepared_betas")
            else:
                fmri_root = os.path.join(args.root, args.dataset_name, "derivatives", "GLM", subjectID_original, "prepared_betas")
            beta_file = os.path.join(fmri_root, f"{subjectID_original}_organized_betas_task-{task}_normalized.pkl")
            if not os.path.isfile(beta_file):
                if task == 'test' or task == 'train':
                    raise ValueError(f"Could not find {task} betas.")
                continue #skips cases where there is no artificial split
            else:
                with open(beta_file, 'rb') as f:
                    betas, stimorder = pickle.load(f)
            numstim, numreps, numvertices = betas.shape
            assert(numstim == len(stimorder))
            assert(numvertices == 91282)
            if task == 'test' or task == 'artificial':
                phase = 'test'
            elif task == 'train':
                phase = 'train'
            for stim in range(numstim):
                for rep in range(numreps):
                    response = betas[stim, rep, :]
                    assert(response.shape[0] == 91282)
                    if np.isnan(response).all(): #all nan responses were filler responses just to keep the matrix dimensionality
                        continue
                    stim_name = Path(stimorder[stim]).stem

                    #identify nans
                    not_real_mask = np.isnan(response) | np.isinf(response) #True means its not real
                    nan_indices = np.where(not_real_mask)[0] #wrt the fsLR32k whole brain space
                    nan_indices_all.update(nan_indices)

                    #add to hdf5 file
                    dset = grp_betas.create_dataset(f"{subjectID}_{fmri_dataset_name}_stimulus-{stim_name}_phase-{phase}_rep-{rep}",
                                                    data=response,
                                                    track_order=True)
                    dset.attrs.create('nan_indices', nan_indices)
                    dset.attrs.create('phase', phase)
                    dset.attrs.create('repetition', rep)
                    dset.attrs.create('presented_stimulus_filename', Path(stimorder[stim]).name)
        #add all nan_indices
        grp.create_dataset('nan_indices_all', data=np.array(list(nan_indices_all))) #these nan indices are not ordered

        #grp.attrs.create('nan_indices_all', np.array(list(nan_indices_all))) 
if __name__=='__main__':
    root_default = os.getenv("DATASETS_ROOT", "/default/path/to/datasets") #use default if DATASETS_ROOT env variable is not set.

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
        participant_info = pd.read_table(os.path.join(args.root, "MOSAIC", "participants", short_to_long[dataset], "participants.tsv"))
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