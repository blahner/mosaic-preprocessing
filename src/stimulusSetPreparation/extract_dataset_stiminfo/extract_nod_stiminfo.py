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
import pickle

"""
In nod the imagenet stim are shown once, not shared across the 30 subjects.
The 120 coco stim are shown 10x for sub 01 and 11x for sub 02-09, shared across subjects.
subs 1-9 saw 4000 imagenet stim and the coco stim. subs 10-30 just saw 1000 imagenet stim.
We defined train and test stim for each subject that includes both imagenet and coco stim, where applicable.
This script defines a test/train split for each subject and how many times each subject saw a stimulus.
The output is a single file with four columns:
- filename (the filename of the stimuli)
- alias (the filename of the stimuli if used a different name in the experiment. n/a for things)
- source (the image source (coco, imagenet))
- test_train (test or train depending on if the video is in the test or train split)
- sub-XX_reps (how many times that subject saw that video)
"""

def main(args):
    save_root = os.path.join(args.dataset_root, "NaturalObjectDataset", "derivatives", "stimuli_metadata")
    if not os.path.exists(save_root):
        raise ValueError(f"save root {save_root} doesnt exist but it should.")
    mosaic_root = os.path.join(args.dataset_root, "MOSAIC", "stimuli", "datasets_stiminfo")
    if not os.path.exists(mosaic_root):
        raise ValueError(f"mosaic root {mosaic_root} doesnt exist but it should.")
    
    cols = ['filename', 'alias', 'source', 'test_train']
    cols.extend([f"sub-{s:02}_reps" for s in range(1,31)])

    filename = []
    alias = []
    img_source = []
    test_train = []
    for sub in range(1,31):
        subject = f"sub-{sub:02}"
        #load imagenet test/train splits per subject that we defined in a previous step since the original authors did not define it.
        with open(os.path.join(save_root, "testtrain_split", f"{subject}_imagenet_groupings_rdm.pkl"), 'rb') as f:
            imagenet_groupings = pickle.load(f)
        for key,values in imagenet_groupings.items():
            for v in values:
                filename.append(f"{v}.JPEG") #all imagenet images have .JPEG extension
                alias.append('n/a')
                img_source.append('ImageNet')
                if key == 'group_01':
                    test_train.append('train')
                elif key == 'group_02':
                    test_train.append('test')
                else:
                    raise ValueError(f"Group {key} not recognized")

    #load the coco test/train splits that we defined in a previous step since the original authors did not define it.
    with open(os.path.join(save_root, "testtrain_split", "coco_groupings_rdm.pkl"), 'rb') as f:
        coco_groupings = pickle.load(f)
    for key,values in coco_groupings.items():
        for v in values:
            filename.append(f"{v}.jpg") #all coco images have .jpg extension
            alias.append('n/a')
            img_source.append('COCO')
            if key == 'group_01':
                test_train.append('train')
            elif key == 'group_02':
                test_train.append('test')
            else:
                raise ValueError(f"Group {key} not recognized")
    
    df = pd.DataFrame(np.zeros((len(filename), len(cols))).astype(int), columns=cols)       
    #fill in the columns we already know wihtout the events file
    df['filename'] = filename
    df['alias'] = alias
    df['source'] = img_source
    df['test_train'] = test_train

    for sub in range(1,31):
        subject = f"sub-{int(sub):02}"
        session_path = os.path.join(args.dataset_root, "derivatives", "fmriprep", subject)
        sessions_tmp = sorted(glob.glob(os.path.join(session_path, f"*imagenet*"))) + sorted(glob.glob(os.path.join(session_path, f"*coco*")))  #compile all the session numbers
        assert(len(sessions_tmp) > 0)
        sessions = []
        for s in sessions_tmp:
            sname = s.split("/")[-1]
            if "imagenet05" not in sname:
                sessions.append(sname)
        assert(len(sessions) > 0)
        if args.verbose:
            print(f"Found {len(sessions)} sessions for subject {subject}")
            print(f"{sessions}")
        for session_path in sessions:
            session = session_path.split('/')[-1]
            if 'coco' in session:
                task='coco'
                if subject == 'sub-01':
                    events_stim_field = 'stim_file'
                else:
                    events_stim_field = 'condition'
            elif 'imagenet' in session:
                task='imagenet'
                events_stim_field = 'stim_file'
            else:
                raise ValueError("Invalid task name. Must be either coco or imagenet session.")

            numruns = len(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-*_desc-confounds_timeseries.tsv")))  #nothing special about the confounds file choice
            assert(numruns > 0)
            if args.verbose:
                print(f"Found {numruns} runs for subject {subject} session {session}")

            ##Load eventts and data for each run
            for run in range(1,numruns+1):
                tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
                for stim_id in tmp.loc[:,events_stim_field]:
                    if str(stim_id) == 'nan': #for coco subject 1, blank trials are not input into the events file. for subjects 2-9, they are listed as n/a conditions with their own onsets
                        continue
                    if task == 'imagenet':
                        stim_parts = stim_id.split('/')
                        fname = stim_parts[-1]
                    elif task == 'coco':
                        stim_parts = stim_id.split('/')
                        fname = stim_parts[-1]
                    df.loc[df['filename'] == fname, f"sub-{sub:02}_reps"] += 1

    assert(df['filename'].nunique() == 57120) #total unique videos
    assert((df['test_train'] == 'test').sum() == 11424) #number in test set
    assert((df['test_train'] == 'train').sum() == 45696) #number in test set
    assert((df['source'] == 'COCO').sum() == 120) 
    assert((df['source'] == 'ImageNet').sum() == 57000) 

    subject_rep_columns = [col for col in df.columns if col.startswith('sub-') and col.endswith('_reps')]

    df_test = df[df['test_train'] == 'test']
    total_reps_test = df_test[subject_rep_columns].sum().sum()
    assert total_reps_test == 13752, f"Total repetitions {total_reps_test} do not match the expected value of 13752"
    
    df_train = df[df['test_train'] == 'train']
    total_reps_train = df_train[subject_rep_columns].sum().sum()
    assert total_reps_train == 55008, f"Total repetitions {total_reps_train} do not match the expected value of 55008"

    if args.verbose:
        print("saving stimulus info file...")
    df.to_csv(os.path.join(save_root, "nod_stiminfo.tsv"), sep='\t', index=False)
    df.to_csv(os.path.join(mosaic_root, "nod_stiminfo.tsv"), sep='\t', index=False)

if __name__ == '__main__':
    dataset_root_default = os.getenv("DATASETS_ROOT", "/default/path/to/datasets") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
