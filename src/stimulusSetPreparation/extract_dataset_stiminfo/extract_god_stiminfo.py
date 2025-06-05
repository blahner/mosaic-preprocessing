from dotenv import load_dotenv
load_dotenv()
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
import glob
import argparse
import numpy as np

"""
In GOD the training stim are shown once, shared across the five subjects.
The testing stim are shown 35 times to each subject, shared across subjects. Train and test stim are in their own sessions.
This script defines a test/train split for each subject and how many times each subject saw a stimulus.
The output is a single file with four columns:
- filename (the filename of the stimuli)
- alias (the filename of the stimuli if used a different name in the experiment)
- source (the source of the image. ImageNet for all)
- test_train (test or train depending on if the video is in the test or train split)
- sub-XX_reps (how many times that subject saw that video)
"""
       
def main(args):
    save_root = os.path.join(args.dataset_root, "GenericObjectDecoding", "derivatives", "stimuli_metadata")
    if not os.path.exists(save_root):
        raise ValueError(f"save root {save_root} doesnt exist but it should.")
    mosaic_root = os.path.join(args.dataset_root, "MOSAIC", "stimuli", "datasets_stiminfo")
    if not os.path.exists(mosaic_root):
        raise ValueError(f"mosaic root {mosaic_root} doesnt exist but it should.")
    
    cols = ['filename', 'alias', 'source', 'test_train']
    cols.extend([f"sub-{s:02}_reps" for s in range(1,6)])
    task = 'perception'

    test_map = pd.read_csv(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "images", "image_test_id.csv"), names=['stim_id', 'filename'])
    train_map = pd.read_csv(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "images", "image_training_id.csv"), names=['stim_id', 'filename'])   

    filename = []
    source = []
    alias = []
    test_train = []
    for sub in range(1,6):
        subject = f"sub-{int(sub):02}"
        print(f"starting {subject}")
        sessions = sorted(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, f"ses-{task}*"))) #compile all the session numbers that have at least one run from this task
        assert(len(sessions) > 0)
        session_name = []
        for s in sessions:
            session_name.append(s.split("/")[-1])
        assert(len(session_name) > 0)
        for session in session_name:
            numruns = len(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-*_desc-confounds_timeseries.tsv")))  #nothing special about the confounds file choice
            assert(numruns > 0)
            if args.verbose:
                print(f"Found {numruns} runs for subject {subject} session {session}")
    
            for run in range(1,numruns+1):
                tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
                stim_id_prev = 0
                for stim_id in tmp.loc[:,'stim_id']:
                    if np.isnan(stim_id) or (stim_id == stim_id_prev):
                        stim_id_prev = stim_id
                        continue #dont include the resting scans at beginning or end or the one-back repeated stimuli
                    if 'Train' in session:
                        stim = train_map.loc[train_map['stim_id'] == stim_id, 'filename'].to_list()[0]
                    elif 'Test' in session:
                        stim = test_map.loc[test_map['stim_id'] == stim_id, 'filename'].to_list()[0]
                    else:
                        raise ValueError(f"session {session} not part of train or test.")
                    if stim not in filename:
                        filename.append(stim)
                        alias.append(stim_id)
                        source.append("ImageNet") #all GOD images are from imagenet
                        if 'Train' in session:
                            test_train.append('train')
                        elif 'Test' in session:
                            test_train.append('test')
                        else:
                            raise ValueError(f"session {session} not part of train or test.")
                    stim_id_prev = stim_id

    df = pd.DataFrame(np.zeros((len(filename), len(cols))).astype(int), columns=cols)       
    #fill in the columns we already know wihtout the events file
    df['filename'] = filename
    df['alias'] = alias
    df['source'] = source
    df['test_train'] = test_train

    for sub in range(1,6):
        subject = f"sub-{int(sub):02}"
        print(f"starting {subject}")
        sessions = sorted(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, f"ses-{task}*"))) #compile all the session numbers that have at least one run from this task
        assert(len(sessions) > 0)
        session_name = []
        for s in sessions:
            session_name.append(s.split("/")[-1])
        assert(len(session_name) > 0)
        for session in session_name:
            numruns = len(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-*_desc-confounds_timeseries.tsv")))  #nothing special about the confounds file choice
            assert(numruns > 0)
            if args.verbose:
                print(f"Found {numruns} runs for subject {subject} session {session}")
    
            ##Load eventts and data for each run
            for run in range(1,numruns+1):
                tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
                stim_id_prev = 0
                for stim_id in tmp.loc[:,'stim_id']:
                    if np.isnan(stim_id) or (stim_id == stim_id_prev):
                        stim_id_prev = stim_id
                        continue #dont include the resting scans at beginning or end or the one-back repeated stimuli
                    if 'Train' in session:
                        stim = train_map.loc[train_map['stim_id'] == stim_id, 'filename'].to_list()[0]
                    elif 'Test' in session:
                        stim = test_map.loc[test_map['stim_id'] == stim_id, 'filename'].to_list()[0]
                    else:
                        raise ValueError(f"session {session} not part of train or test.")
                    df.loc[df['filename'] == stim, f"{subject}_reps"] += 1
                    stim_id_prev = stim_id

    assert(df['filename'].nunique() == 1250) #total unique stimuli
    assert (df['test_train'] == 'train').sum() == 1200, f"Total repetitions {total_reps_train} do not match the expected value of 6000"
    assert (df['test_train'] == 'test').sum() == 50, f"Total repetitions {total_reps_train} do not match the expected value of 6000"
    subject_rep_columns = [col for col in df.columns if col.startswith('sub-') and col.endswith('_reps')]

    df_test = df[df['test_train'] == 'test']
    total_reps_test = df_test[subject_rep_columns].sum().sum()
    assert total_reps_test == 8750, f"Total repetitions {total_reps_test} do not match the expected value of 8750"
    
    df_train = df[df['test_train'] == 'train']
    total_reps_train = df_train[subject_rep_columns].sum().sum()
    assert total_reps_train == 6000, f"Total repetitions {total_reps_train} do not match the expected value of 6000"

    if args.verbose:
        print("saving stimulus info file...")
    df.to_csv(os.path.join(save_root, "god_stiminfo.tsv"), sep='\t', index=False)
    df.to_csv(os.path.join(mosaic_root, "god_stiminfo.tsv"), sep='\t', index=False)

if __name__ == '__main__':
    dataset_root_default = os.getenv("DATASETS_ROOT", "/default/path/to/datasets") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
