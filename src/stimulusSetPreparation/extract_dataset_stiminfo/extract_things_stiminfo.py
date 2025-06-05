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
In THINGS the training stim are shown once, shared across the three subjects.
The testing stim are shown 12 times to each subject, shared across subjects. Train and test stim are interspersed.
This script defines a test/train split for each subject and how many times each subject saw a stimulus.
The output is a single file with four columns:
- filename (the filename of the stimuli)
- alias (the filename of the stimuli if used a different name in the experiment. n/a for things)
- test_train (test or train depending on if the video is in the test or train split)
- sub-XX_reps (how many times that subject saw that video)
"""
       
def main(args):
    save_root = os.path.join(args.dataset_root, "THINGS_fmri", "derivatives", "stimuli_metadata")
    if not os.path.exists(save_root):
        raise ValueError(f"save root {save_root} doesnt exist but it should.")
    mosaic_root = os.path.join(args.dataset_root, "MOSAIC", "stimuli", "datasets_stiminfo")
    if not os.path.exists(mosaic_root):
        raise ValueError(f"mosaic root {mosaic_root} doesnt exist but it should.")
    
    cols = ['filename', 'alias', 'source', 'test_train']
    cols.extend([f"sub-{s:02}_reps" for s in range(1,4)])
    task = 'things'
    numsessions = 12 #all subjects saw 12 sessions
    filename = []
    alias = []
    source = []
    test_train = []
    for sub in range(1,4):
        subject = f"sub-{int(sub):02}"
        print(f"starting {subject}")
        for ses in range(1,numsessions+1):
            session = f"ses-{task}{ses:02}"
            numruns = len(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-*_desc-confounds_timeseries.tsv"))) 
            for run in range(1,numruns+1):
                tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
                for idx, img_filename in enumerate(tmp.loc[:,'file_path']):
                    if tmp.loc[idx,'trial_type'] == 'catch':
                        continue
                    cat, fname = img_filename.split('/')
                    if "b.jpg" in fname:
                        img_source = 'behavior'
                    elif "n.jpg" in fname:
                        img_source = 'ImageNet'
                    elif "s.jpg" in fname:
                        img_source = 'googleimages'
                    else:
                        raise ValueError("Source not found.")
                    if (tmp.loc[idx,'trial_type'] == 'exp') and (fname not in filename):
                        test_train.append('train')
                        filename.append(fname)
                        alias.append('n/a')
                        source.append(img_source)
                        
                    elif (tmp.loc[idx,'trial_type'] == 'test') and (fname not in filename):
                        test_train.append('test')
                        filename.append(fname)
                        alias.append('n/a')
                        source.append(img_source)

    df = pd.DataFrame(np.zeros((len(filename), len(cols))).astype(int), columns=cols)       
    #fill in the columns we already know wihtout the events file
    df['filename'] = filename
    df['alias'] = alias
    df['test_train'] = test_train
    df['source'] = source

    for sub in range(1,4):
        subject = f"sub-{int(sub):02}"
        print(f"starting {subject}")
        for ses in range(1,numsessions+1):
            session = f"ses-{task}{int(ses):02}"
            numruns = len(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-*_desc-confounds_timeseries.tsv")))  #nothing special about the confounds file choice
            assert(numruns > 0)
            if args.verbose:
                print(f"Found {numruns} runs for subject {subject} session {session}")
            ##Load eventts and data for each run
            for run in range(1,numruns+1):
                #load events
                tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
                for idx, img_filename in enumerate(tmp.loc[:,'file_path']):
                    if tmp.loc[idx,'trial_type'] == 'catch':
                        continue
                    _, fname = img_filename.split('/')
                    df.loc[df['filename'] == fname, f"{subject}_reps"] += 1

    assert(df['filename'].nunique() == 8740) #total unique videos
    assert((df['test_train'] == 'test').sum() == 100) #number in test set
    assert((df['test_train'] == 'train').sum() == 8640) #number in test set
    subject_rep_columns = [col for col in df.columns if col.startswith('sub-') and col.endswith('_reps')]

    df_test = df[df['test_train'] == 'test']
    total_reps_test = df_test[subject_rep_columns].sum().sum()
    assert total_reps_test == 3600, f"Total repetitions {total_reps_test} do not match the expected value of 3600"
    
    df_train = df[df['test_train'] == 'train']
    total_reps_train = df_train[subject_rep_columns].sum().sum()
    assert total_reps_train == 25920, f"Total repetitions {total_reps_train} do not match the expected value of 25920"

    if args.verbose:
        print("saving stimulus info file...")
    df.to_csv(os.path.join(save_root, "things_stiminfo.tsv"), sep='\t', index=False)
    df.to_csv(os.path.join(mosaic_root, "things_stiminfo.tsv"), sep='\t', index=False)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"THINGS_fmri") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
