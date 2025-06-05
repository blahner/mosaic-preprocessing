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
import re
"""
In BOLD5000 the training stim are shown once, not shared across the three subjects.
The testing stim are shown 1-5 times to each subject, shared across subjects. Train and test stim are interspersed.
This script defines a test/train split for each subject and how many times each subject saw a stimulus.
The output is a single file with four columns:
- filename (the filename of the stimuli)
- alias (the filename of the stimuli if used a different name in the experiment. n/a for things)
- source (the image source (SUN, coco, imagenet))
- test_train (test or train depending on if the video is in the test or train split)
- sub-XX_reps (how many times that subject saw that video)
"""

def classify_image(filename):
    # Define regex patterns for each group
    imagenet_pattern = r'^n\d{8}_\d+\.(jpg|jpeg|JPEG|JPG)$'
    coco_pattern = r'^\d+\.(jpg|jpeg|JPEG|JPG)$'
    sun_pattern = r'^[a-zA-Z_]+\d*\.(jpg|jpeg|JPEG|JPG)$'
    
    # Match the filename against each pattern
    if re.match(imagenet_pattern, filename):
        return 'ImageNet'
    elif re.match(coco_pattern, filename):
        return 'COCO'
    elif re.match(sun_pattern, filename):
        return 'SUN'
    else:
        return 'Unknown'

def main(args):
    save_root = os.path.join(args.dataset_root, "BOLD5000", "derivatives", "stimuli_metadata")
    if not os.path.exists(save_root):
        raise ValueError(f"save root {save_root} doesnt exist but it should.")
    mosaic_root = os.path.join(args.dataset_root, "MOSAIC", "stimuli", "datasets_stiminfo")
    if not os.path.exists(mosaic_root):
        raise ValueError(f"mosaic root {mosaic_root} doesnt exist but it should.")
    
    subject_numsessions = {"sub-CSI1": 15,
                      "sub-CSI2": 15,
                      "sub-CSI3": 15,
                      "sub-CSI4": 9}
    cols = ['filename', 'alias', 'source', 'test_train']
    cols.extend([f"sub-{s:02}_reps" for s in range(1,5)])
    task="5000scenes"
    test_stimuli_all_tmp = pd.read_csv(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "Scene_Stimuli", "repeated_stimuli_113_list.txt"), header=None).values.flatten().tolist()
    test_stimuli_all = [stim.replace('COCO_train2014_','') for stim in test_stimuli_all_tmp]
    filename = []
    alias = []
    img_source = []
    test_train = []
    for sub in range(1,5):
        subject = f"sub-CSI{int(sub)}"
        print(f"starting {subject}")
        numsessions = subject_numsessions[subject]
        for session_num in range(1, numsessions+1):
            session = f"ses-{int(session_num):02}"

            sub_func_root = os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func")
            numruns = len(glob.glob(os.path.join(sub_func_root, f"*task-{task}_*confounds_timeseries.tsv")))
            assert(numruns > 0)
            ##Load events and data for each run
            for run in range(1,numruns+1):
                if args.verbose:
                    print("run:",run)
                tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
                for fname in tmp.loc[:,'stim_file']:
                    fname_noprefix = fname.replace('COCO_train2014_','') #removes the COCO prefix if it exists
                    if fname_noprefix not in filename:
                        filename.append(fname_noprefix)
                        source = classify_image(fname_noprefix)
                        img_source.append(source)
                        if source == 'ImageNet':
                            cat = fname_noprefix.split('_')[0]
                        else:
                            cat = 'n/a'
                        if source == 'COCO':
                            alias.append(fname) #with the COCO_train2014_ prefix
                        else:
                            alias.append('n/a')
                        if (fname_noprefix in test_stimuli_all):
                            test_train.append('test')
                        else:
                            test_train.append('train')

    df = pd.DataFrame(np.zeros((len(filename), len(cols))).astype(int), columns=cols)       
    #fill in the columns we already know wihtout the events file
    df['filename'] = filename
    df['alias'] = alias
    df['source'] = img_source
    df['test_train'] = test_train

    for sub in range(1,5):
        subject = f"sub-CSI{int(sub)}"
        if args.verbose:
            print(f"starting {subject}")
        numsessions = subject_numsessions[subject]
        for session_num in range(1, numsessions+1):
            session = f"ses-{int(session_num):02}"

            sub_func_root = os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func")
            numruns = len(glob.glob(os.path.join(sub_func_root, f"*task-{task}_*confounds_timeseries.tsv")))
            assert(numruns > 0)
            ##Load events and data for each run
            for run in range(1,numruns+1):
                tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
                for fname in tmp.loc[:,'stim_file']:
                    fname_noprefix = fname.replace('COCO_train2014_','') #removes the COCO prefix if it exists
                    df.loc[df['filename'] == fname_noprefix, f"sub-{sub:02}_reps"] += 1

    assert(df['filename'].nunique() == 4916) #total unique videos
    assert((df['test_train'] == 'test').sum() == 113) #number in test set
    assert((df['test_train'] == 'train').sum() == 4803) #number in test set
    assert((df['source'] == 'SUN').sum() == 1000)
    assert((df['source'] == 'COCO').sum() == 2000) 
    assert((df['source'] == 'ImageNet').sum() == 1916) 

    subject_rep_columns = [col for col in df.columns if col.startswith('sub-') and col.endswith('_reps')]

    df_test = df[df['test_train'] == 'test']
    total_reps_test = df_test[subject_rep_columns].sum().sum()
    assert total_reps_test == 1615, f"Total repetitions {total_reps_test} do not match the expected value of 1,615"
    
    df_train = df[df['test_train'] == 'train']
    total_reps_train = df_train[subject_rep_columns].sum().sum()
    assert total_reps_train == 17255, f"Total repetitions {total_reps_train} do not match the expected value of 17255"

    if args.verbose:
        print("saving stimulus info file...")
    df.to_csv(os.path.join(save_root, "b5000_stiminfo.tsv"), sep='\t', index=False)
    df.to_csv(os.path.join(mosaic_root, "b5000_stiminfo.tsv"), sep='\t', index=False)

if __name__ == '__main__':
    dataset_root_default = os.getenv("DATASETS_ROOT", "/default/path/to/datasets") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
