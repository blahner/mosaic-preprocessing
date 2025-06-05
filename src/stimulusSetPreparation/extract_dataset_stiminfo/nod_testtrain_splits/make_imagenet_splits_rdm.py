from dotenv import load_dotenv
load_dotenv()
import os
import argparse
import random
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_lowertriangular(rdm):
    num_conditions = rdm.shape[0]
    return rdm[np.triu_indices(num_conditions,1)]

def main(args):
    #set random seed
    random.seed(42)

    model_name = 'dreamsim'
    task='imagenet'
    events_stim_field = 'stim_file'
    traintest_ratio = 0.8 

    for sub in range(1,31):
        subject = f"sub-{int(sub):02}"
        print(f"starting {subject}")
        session_path = os.path.join(args.dataset_root, "derivatives", "fmriprep", subject)
        sessions_tmp = sorted(glob.glob(os.path.join(session_path, f"*{task}*"))) #compile all the session numbers
        assert(len(sessions_tmp) > 0)
        sessions = []
        for s in sessions_tmp:
            sname = s.split("/")[-1]
            if "imagenet05" not in sname:
                sessions.append(sname)
        assert(len(sessions) > 0)
        print(f"Found {len(sessions)} sessions")
        print(f"{sessions}")
        allsession_conds = [] #collects the filenames of all imagenet conditions shown to the subject
        for session_path in sessions:
            session = session_path.split('/')[-1]
            numruns = len(glob.glob(os.path.join(args.dataset_root, "derivatives", "fmriprep", subject, session, "func", f"{subject}_{session}_task-{task}_run-*_desc-confounds_timeseries.tsv")))  #nothing special about the confounds file choice
            assert(numruns > 0)
            print(f"Found {numruns} runs for subject {subject} session {session}")

            ##Load eventts and data for each run
            for _, run in enumerate(range(1,numruns+1)):
                print("run:",run)

                #load events
                tmp = pd.read_table(os.path.join(args.dataset_root, "Nifti", subject, session, "func", f"{subject}_{session}_task-{task}_run-{run:02}_events.tsv"))
                for stim_id in tmp.loc[:,events_stim_field]:
                    if str(stim_id) == 'nan': #for coco subject 1, blank trials are not input into the events file. for subjects 2-9, they are listed as n/a conditions with their own onsets
                        continue
                    stim_parts = stim_id.split('/')[-1]
                    imagenet_filename = stim_parts.split('.')[0]
                    if imagenet_filename not in allsession_conds:
                        allsession_conds.append(imagenet_filename)
        if sub < 10:
            assert(len(allsession_conds) == 4000) #subjects 1-9 saw 4000 unique imagenet images, subjects 10-30 saw 1000 unique imagenet images
        elif sub >= 10:
            assert(len(allsession_conds) == 1000) #subjects 1-9 saw 4000 unique imagenet images, subjects 10-30 saw 1000 unique imagenet images

        #load embeddings
        numconditions = len(allsession_conds)
        embedding_paths = sorted(glob.glob(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", f"dreamsim_embeddings", f"n*_model-{model_name}.npy")))
        embeddings = []
        image_filename = []
        for embedding_path in embedding_paths:
            filename = embedding_path.split('/')[-1]
            imagenet_filename = filename.split(f'_model-{model_name}.npy')[0]
            if imagenet_filename in allsession_conds:
                embeddings.append(np.load(embedding_path))
                image_filename.append(imagenet_filename)
            else:
                continue

        assert(len(embeddings) == numconditions)

        rdm = 1 - cosine_similarity(np.array(embeddings))
        assert(rdm.shape == (numconditions,numconditions))

        distance = []
        for row in range(numconditions):
            indices = np.ones((numconditions,))
            indices[row] = 0 #make the identity index 0
            d_tmp = rdm[row,indices.astype(bool)]
            assert(len(d_tmp) == numconditions-1)
            distance.append(np.median(d_tmp))

        #find the indices of the top X values (largest distance)
        sorted_idx = np.argsort(distance)
        train_indices = sorted_idx[:int(numconditions*traintest_ratio)]
        test_indices = sorted_idx[int(numconditions*traintest_ratio):]
        data = {'group_01': [], 'group_02': []}
        for idx, stim in enumerate(image_filename):
            if idx in train_indices:
                data['group_01'].append(stim)
            elif idx in test_indices:
                data['group_02'].append(stim)
            else:
                raise ValueError(f"idx {idx} not found in test or train")
        #save groups
        print("saving groupings...")
        with open(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "testtrain_split", f"{subject}_imagenet_groupings_rdm.pkl"), 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"NaturalObjectDataset") #use default if DATASETS_ROOT env variable is not set.
    project_root_default = os.getenv("PROJECT_ROOT", "/default/path/to/datasets") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-p", "--project_root", default=project_root_default, help="The root path to the project directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)