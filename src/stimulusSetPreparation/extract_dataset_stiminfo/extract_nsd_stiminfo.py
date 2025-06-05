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
from pycocotools.coco import COCO

"""
In NSD the training stim are shown 3x, not shared across the 8 subjects.
The testing stim are shown 3x times to each subject, shared across subjects. Train and test stim are interspersed.
This script defines a test/train split for each subject and how many times each subject saw a stimulus.
The output is a single file with four columns:
- filename (the original filename of the stimuli as used in COCO)
- alias (the filename of the stimuli if used a different name in the experiment. this is the nsdID for nsd.)
- source (the image source. COCO for all nsdcore.)
- test_train (test or train depending on if the video is in the test or train split)
- sub-XX_reps (how many times that subject saw that video)
"""

def ceildiv(a, b):
    return -(a // -b)

def main(args):
    save_root = os.path.join(args.dataset_root, "NaturalScenesDataset", "derivatives", "stimuli_metadata")
    if not os.path.exists(save_root):
        raise ValueError(f"save root {save_root} doesnt exist but it should.")
    mosaic_root = os.path.join(args.dataset_root, "MOSAIC", "stimuli", "datasets_stiminfo")
    if not os.path.exists(mosaic_root):
        raise ValueError(f"mosaic root {mosaic_root} doesnt exist but it should.")
    
    session_info = {'sub-01': 40, 'sub-02': 40, 'sub-03': 32, 'sub-04': 30,
                'sub-05': 40, 'sub-06': 32, 'sub-07': 40, 'sub-08': 30} #number of nsdcore sessions per subject
    cols = ['filename', 'alias', 'source', 'test_train']
    cols.extend([f"sub-{s:02}_reps" for s in range(1,9)])
    task="nsdcore"

    #load stimuli info
    annotations_root = os.path.join(args.dataset_root,"NaturalScenesDataset", "derivatives", "stimuli_metadata")
    nsd_csv = pd.read_csv(os.path.join(annotations_root, "nsd_stim_info_merged.csv"))
    shared1000_nsdId = nsd_csv.loc[nsd_csv['shared1000'], 'nsdId'].values #these nsdId values are 0-indexed
    assert(len(shared1000_nsdId) == 1000)
    coco_annotation_val = COCO(annotation_file=os.path.join(annotations_root, "annotations_trainval2017", "annotations", "instances_val2017.json"))
    coco_annotation_train = COCO(annotation_file=os.path.join(annotations_root, "annotations_trainval2017", "annotations", "instances_train2017.json"))
    img_ids_val = coco_annotation_val.getImgIds()
    img_ids_train = coco_annotation_train.getImgIds()

    filename = []
    alias = []
    source = []
    test_train = []
       
    for sub in range(1,9):
        subject = f"sub-{sub:02}"
        numsessions = session_info[subject]
        for ses in range(1, numsessions+1):
            session = f"ses-nsd{ses:02}"
            print(f"{subject} {session}")
            numruns = len(glob.glob(os.path.join(args.dataset_root, "NaturalScenesDataset", "Nifti", subject, session, "func", f"*_task-{task}_*_events.tsv")))
            #print(f"numruns {numruns}")
            for r in range(1,numruns+1):
                tmp = pd.read_table(os.path.join(args.dataset_root, "NaturalScenesDataset", "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{r:02}_events.tsv"))
                for stim_id in tmp.loc[:,'73k_id']:
                    #nsd 73k id to coco id
                    cocoID = nsd_csv.loc[nsd_csv['nsdId'] == (stim_id-1), 'cocoId'].to_list()[0] #the stim_id in the event file is 1-indexed but the nsdId in the "nsd_stim_info_merged.csv" is 0-indexed
                    #coco id to image filename
                    if cocoID in img_ids_val:
                        img_info = coco_annotation_val.loadImgs(cocoID)[0]
                    elif cocoID in img_ids_train:
                        img_info = coco_annotation_train.loadImgs(cocoID)[0]
                    else:
                        raise ValueError(f"image coco id {cocoID} not found.")
                    stim = img_info['file_name']
                    if stim not in filename:
                        filename.append(stim)
                        alias.append((stim_id-1))
                        source.append("COCO")
                        if (stim_id-1) in shared1000_nsdId:
                            test_train.append('test')
                        else:
                            test_train.append('train')

        #now load the synthetic stimuli for each subject
        session = 'ses-nsdsynthetic'
        numruns_synthetic = 8
        for run in range(1,numruns_synthetic+1):
            if run%2 == 0: #if run is even
                synthetic_task = 'memory' #or one-back
            else:
                synthetic_task = 'fixation'
            synthetic_task_run = ceildiv(run, 2)
            #load events. just getting the filenames so dont have to worry about one-back repeats
            tmp = pd.read_table(os.path.join(args.dataset_root, "NaturalScenesDataset", "Nifti", subject, session, "func", f"{subject}_{session}_task-{synthetic_task}_run-{synthetic_task_run:02}_events.tsv"))
            for stimIDX in tmp.loc[:,'stim_idx']:
                stim = f"{stimIDX:03}.jpg"
                if stim not in filename:
                    filename.append(stim)
                    alias.append(stimIDX)
                    source.append("NSDsynthetic")
                    test_train.append("test")

    df = pd.DataFrame(np.zeros((len(filename), len(cols))).astype(int), columns=cols)       
    #fill in the columns we already know wihtout the events file
    df['filename'] = filename
    df['alias'] = alias
    df['source'] = source
    df['test_train'] = test_train

    for sub in range(1,9):
        subject = f"sub-{sub:02}"
        numsessions = session_info[subject]
        for ses in range(1, numsessions+1):
            session = f"ses-nsd{ses:02}"
            print(f"{subject} {session}")
            numruns = len(glob.glob(os.path.join(args.dataset_root,  "NaturalScenesDataset","Nifti", subject, session, "func", f"*_task-{task}_*_events.tsv")))
            #print(f"numruns {numruns}")
            for r in range(1,numruns+1):
                tmp = pd.read_table(os.path.join(args.dataset_root, "NaturalScenesDataset", "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{r:02}_events.tsv"))
                for stim_id in tmp.loc[:,'73k_id']:
                    #nsd 73k id to coco id
                    cocoID = nsd_csv.loc[nsd_csv['nsdId'] == (stim_id-1), 'cocoId'].to_list()[0] #the stim_id in the event file is 1-indexed but the nsdId in the "nsd_stim_info_merged.csv" is 0-indexed
                    #coco id to image filename
                    if cocoID in img_ids_val:
                        img_info = coco_annotation_val.loadImgs(cocoID)[0]
                    elif cocoID in img_ids_train:
                        img_info = coco_annotation_train.loadImgs(cocoID)[0]
                    else:
                        raise ValueError(f"image coco id {cocoID} not found.")
                    stim = img_info['file_name']
                    df.loc[df['filename'] == stim, f"{subject}_reps"] += 1

    for sub in range(1,9):
        subject = f"sub-{sub:02}"
        #now load the synthetic stimuli for each subject
        session = 'ses-nsdsynthetic'
        numruns_synthetic = 8
        for run in range(1,numruns_synthetic+1):
            if run%2 == 0: #if run is even
                synthetic_task = 'memory' #or one-back
            else:
                synthetic_task = 'fixation'
            synthetic_task_run = ceildiv(run, 2)
            
            #load events
            tmp = pd.read_table(os.path.join(args.dataset_root, "NaturalScenesDataset", "Nifti", subject, session, "func", f"{subject}_{session}_task-{synthetic_task}_run-{synthetic_task_run:02}_events.tsv"))
            prev_stim = np.nan
            for stimIDX in tmp.loc[:,'stim_idx']:
                if stimIDX == prev_stim:
                    continue #skip one-back stimuli
                stim = f"{stimIDX:03}.jpg"
                df.loc[df['filename'] == stim, f"{subject}_reps"] += 1
                prev_stim = stimIDX #update the previous stimuli for next row
            
    assert(df['filename'].nunique() == 70566 + 284) #total unique stimuli
    assert((df['test_train'] == 'test').sum() == 1000 + 284) #number in test set
    assert((df['test_train'] == 'train').sum() == 69566) #number in test set
    assert((df['source'] == 'COCO').sum() == 70566) 
    assert((df['source'] == 'NSDsynthetic').sum() == 284) 

    subject_rep_columns = [col for col in df.columns if col.startswith('sub-') and col.endswith('_reps')]

    df_test = df[(df['test_train'] == 'test') & (df['source'] == 'COCO')]
    total_reps_test = df_test[subject_rep_columns].sum().sum()
    assert total_reps_test == 21118, f"Total repetitions {total_reps_test} do not match the expected value of 21,118"
    
    df_train = df[(df['test_train'] == 'train') & (df['source'] == 'COCO')]
    total_reps_train = df_train[subject_rep_columns].sum().sum()
    assert total_reps_train == 191882, f"Total repetitions {total_reps_train} do not match the expected value of 191,882"

    if args.verbose:
        print("saving stimulus info file...")
    df.to_csv(os.path.join(save_root, "nsd_stiminfo.tsv"), sep='\t', index=False)
    df.to_csv(os.path.join(mosaic_root, "nsd_stiminfo.tsv"), sep='\t', index=False)

    if args.verbose:
        print("saving a list of stimuli that are not naturalistic...")
    df_notnatural = pd.DataFrame(df.loc[df['source'] == 'NSDsynthetic', 'filename'].tolist(), columns=['filename'])
    assert(len(df_notnatural) == 284)
    df_notnatural.to_csv(os.path.join(save_root, "nsd_artificial_list.tsv"), sep='\t', index=False)
    df_notnatural.to_csv(os.path.join(mosaic_root, "nsd_artificial_list.tsv"), sep='\t', index=False)

if __name__ == '__main__':
    dataset_root_default = os.getenv("DATASETS_ROOT", "/default/path/to/datasets") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
