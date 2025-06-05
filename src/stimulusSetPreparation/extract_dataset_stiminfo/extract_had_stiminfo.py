from dotenv import load_dotenv
load_dotenv()
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
import argparse
import numpy as np

"""
For each subject, the videos from the last three runs (10,11,12) compose the test split as the they contain one video from each of the 180 categories.
The other 9 runs are train.
This script defines a test/train split for each subject.
The output is a single file with four columns:
- filename (the filename of the video)
- seen_by (the subject that saw the video)
- run (the run number the video was presented)
- test_train (test or train depending on if the video is in the test or train split)
"""
       
def main(args):
    save_root = os.path.join(args.dataset_root, "HumanActionsDataset", "derivatives", "stimuli_metadata")
    if not os.path.exists(save_root):
        raise ValueError(f"save root {save_root} doesnt exist but it should.")    
    mosaic_root = os.path.join(args.dataset_root, "MOSAIC", "stimuli", "datasets_stiminfo")
    if not os.path.exists(mosaic_root):
        raise ValueError(f"mosaic root {mosaic_root} doesnt exist but it should.")
    
    session = "ses-action01" #HAD only has one session called "action01"
    numruns = 12 #each subject has 12 runs
    all_filenames = set() #just keeps track of filenames to ensure no repeats

    cols = ['filename', 'alias', 'source', 'test_train']
    cols.extend([f"sub-{s:02}_reps" for s in range(1,31)])

    filename = []
    alias = []
    img_source = []
    test_train = []
    for sub in range(1,31):
        subject = f"sub-{sub:02}"
        if args.verbose:
            print(f"Loading subject {subject}")
        for run in range(1,numruns+1):
            #load events
            tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-action_run-{run:02d}_events.tsv"))
            for tt in tmp.loc[:,'stim_file']:
                _, fname_ext = tt.split("/")
                if fname_ext in all_filenames:
                    raise ValueError(f"filename {fname_ext} already found")
                else:
                    all_filenames.add(fname_ext)
                filename.append(fname_ext)
                img_source.append("HumanActionClipsAndSegments")
                alias.append('n/a')
                if run >= 10:
                    test_train.append('test')
                else:
                    test_train.append('train')

    df = pd.DataFrame(np.zeros((len(filename), len(cols))).astype(int), columns=cols)       
    #fill in the columns we already know wihtout the events file
    df['filename'] = filename
    df['alias'] = alias
    df['source'] = img_source
    df['test_train'] = test_train

    all_filenames = set() #keep track of any repeats once more safekeeping
    for sub in range(1,31):
        subject = f"sub-{sub:02}"
        if args.verbose:
            print(f"Loading subject {subject}")
        for run in range(1,numruns+1):
            #load events
            tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-action_run-{run:02d}_events.tsv"))
            for tt in tmp.loc[:,'stim_file']:
                _, fname_ext = tt.split("/")
                if fname_ext in all_filenames:
                    raise ValueError(f"filename {fname_ext} already found")
                else:
                    all_filenames.add(fname_ext)
                df.loc[df['filename'] == fname_ext, f"sub-{sub:02}_reps"] += 1

    assert(df['filename'].nunique() == 21600) #total unique videos
    assert((df['test_train'] == 'test').sum() == 5400) #number in test set
    assert((df['test_train'] == 'train').sum() == 16200) #number in test set
    assert((df['source'] == 'HumanActionClipsAndSegments').sum() == 21600)

    subject_rep_columns = [col for col in df.columns if col.startswith('sub-') and col.endswith('_reps')]

    df_test = df[df['test_train'] == 'test']
    total_reps_test = df_test[subject_rep_columns].sum().sum()
    assert total_reps_test == 5400, f"Total repetitions {total_reps_test} do not match the expected value of 5400"
    
    df_train = df[df['test_train'] == 'train']
    total_reps_train = df_train[subject_rep_columns].sum().sum()
    assert total_reps_train == 16200, f"Total repetitions {total_reps_train} do not match the expected value of 16200"

    if args.verbose:
        print("saving stimulus info file...")
    df.to_csv(os.path.join(save_root, "had_stiminfo.tsv"), sep='\t', index=False)
    df.to_csv(os.path.join(mosaic_root, "had_stiminfo.tsv"), sep='\t', index=False)

if __name__ == '__main__':
    dataset_root_default = os.getenv("DATASETS_ROOT", "/default/path/to/datasets") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
