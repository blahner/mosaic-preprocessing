from dotenv import load_dotenv
load_dotenv()
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
import argparse
import numpy as np
import json

"""
In BMD the training videos are videos 1-1000 with 3 reps for all 10 subjects, and the testing videos are videos 1001-1102 with 10 reps for all 10 subjects. 
This script defines a test/train split for each subject and how many times each subject saw a video.
The output is a single file with four columns:
- filename (the filename of the video from moments in time dataset)
- alias (the filename of the video as used in the experiment e.g., 0001.mp4)
- source (the source of the video, all moments in time)
- test_train (test or train depending on if the video is in the test or train split)
- sub-XX_reps (how many times that subject saw that video)
"""
       
def main(args):
    save_root = os.path.join(args.dataset_root,"BOLDMomentsDataset", "derivatives", "stimuli_metadata")
    if not os.path.exists(save_root):
        raise ValueError(f"save root {save_root} doesnt exist but it should.")
    mosaic_root = os.path.join(args.dataset_root, "MOSAIC", "stimuli", "datasets_stiminfo")
    if not os.path.exists(mosaic_root):
        raise ValueError(f"mosaic root {mosaic_root} doesnt exist but it should.")
    
    #load stimulus annotations to convert XXXX.mp4 filename to original Moments In Time Filename
    annotations = json.load(open(os.path.join(args.dataset_root,"derivatives", "stimuli_metadata", "annotations.json"), 'r'))
    cols = ['filename', 'alias', 'source', 'test_train']
    cols.extend([f"sub-{s:02}_reps" for s in range(1,11)])
    #first define the pandas dataframe
    df = pd.DataFrame(np.zeros((1102, len(cols))).astype(int), columns=cols)

    filename = []
    alias = []
    source = []
    test_train_stimuli = []
    for xxxx_stim in range(1,1103):
        mit_name = annotations[f"{xxxx_stim:04}"]['MiT_filename'].split('/')[-1]
        filename.append(mit_name)
        source.append("MomentsInTime")
        alias.append(f"{xxxx_stim:04}.mp4")
        if xxxx_stim <= 1000:
            test_train_stimuli.append('train')
        elif xxxx_stim >= 1001:
            test_train_stimuli.append('test')

    #fill in the columns we already know wihtout the events file
    df['filename'] = filename
    df['alias'] = alias
    df['source'] = source
    df['test_train'] = test_train_stimuli

    for sub in range(1,11):
        subject = f"sub-{int(sub):02}"
        print(f"starting {subject}")
        for ses in range(2,6):
            session = f"ses-{int(ses):02}"
            for task in ['test', 'train']:
                if task == 'test':
                    numruns = 3
                elif task == 'train':
                    numruns = 10
                for run in range(1,numruns+1):
                    #load events
                    tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run}_events.tsv"))
                    for tt in tmp.loc[:,'stim_file']:
                        if str(tt) == "nan":
                            continue
                        else:
                            fname = tt.split(f"{task}/")[1]
                            stimIDX = int(fname.split(".mp4")[0]) #stimuli index 1001-1102
                            mit_name = annotations[f"{stimIDX:04}"]['MiT_filename'].split('/')[-1]
                            df.loc[df['filename'] == mit_name, f"{subject}_reps"] += 1
    assert(df['filename'].nunique() == 1102) #total unique videos
    assert(df['alias'].nunique() == 1102) #total unique videos
    assert((df['test_train'] == 'test').sum() == 102) #number in test set
    assert((df['test_train'] == 'train').sum() == 1000) #number in test set
    assert((df['source'] == 'MomentsInTime').sum() == 1102) 

    subject_rep_columns = [col for col in df.columns if col.startswith('sub-') and col.endswith('_reps')]

    df_test = df[df['test_train'] == 'test']
    total_reps_test = df_test[subject_rep_columns].sum().sum()
    assert total_reps_test == 10200, f"Total repetitions {total_reps_test} do not match the expected value of 10200"
    
    df_train = df[df['test_train'] == 'train']
    total_reps_train = df_train[subject_rep_columns].sum().sum()
    assert total_reps_train == 30000, f"Total repetitions {total_reps_train} do not match the expected value of 30000"

    if args.verbose:
        print("saving stimulus info file...")
    df.to_csv(os.path.join(save_root, "bmd_stiminfo.tsv"), sep='\t', index=False)
    df.to_csv(os.path.join(mosaic_root, "bmd_stiminfo.tsv"), sep='\t', index=False)

if __name__ == '__main__':
    dataset_root_default = os.getenv("DATASETS_ROOT", "/default/path/to/datasets") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
