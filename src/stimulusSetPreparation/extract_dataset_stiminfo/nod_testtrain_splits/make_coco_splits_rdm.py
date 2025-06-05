from dotenv import load_dotenv
load_dotenv()
import os
import argparse
import random
import glob
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_lowertriangular(rdm):
    num_conditions = rdm.shape[0]
    return rdm[np.triu_indices(num_conditions,1)]

def main(args):
    #set random seed
    random.seed(42)

    model_name = 'dreamsim'

    #load embeddings
    numconditions = 120
    embedding_paths = sorted(glob.glob(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", f"dreamsim_embeddings", f"00*_model-{model_name}.npy")))
    assert(len(embedding_paths) == numconditions)
    embeddings = []
    image_filename = []
    for embedding_path in embedding_paths:
        embeddings.append(np.load(embedding_path))
        filename = embedding_path.split('/')[-1]
        image_filename.append(filename.split(f'_model-{model_name}.npy')[0])

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
    train_indices = sorted_idx[:96]
    test_indices = sorted_idx[96:]
    data = {'group_01': [], 'group_02': []}
    for idx, stim in enumerate(image_filename):
        if idx in train_indices:
            data['group_01'].append(stim)
        elif idx in test_indices:
            data['group_02'].append(stim)
        else:
            raise ValueError(f"idx {idx} not found in test or train")
    #save groups
    with open(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "testtrain_split", "coco_groupings_rdm.pkl"), 'wb') as f:
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