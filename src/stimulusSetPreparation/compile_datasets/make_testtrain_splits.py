from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import os
from tqdm import tqdm
import glob as glob
import numpy as np 
import pickle
import torch
from torch.nn.functional import cosine_similarity
import json
from pathlib import Path
import argparse

def main(args):
    save_root = os.path.join(args.dataset_root, "MOSAIC")
    compiled_stiminfo = pd.read_table(os.path.join(save_root, "stimuli", "datasets_stiminfo", "compiled_dataset_stiminfo.tsv"), low_memory=False)
    compiled_filenames = compiled_stiminfo['filename']
    assert(len(compiled_filenames) == len(set(compiled_filenames)))

    #load artificial filenames. Note that the compiled_dataset_stiminfo.tsv contains the artificial images and some can be excluded. This list is compile
    #any and all articial images that have not been excluded into its artificial test set.
    deeprecon_artificial = pd.read_csv(os.path.join(save_root, "stimuli", "datasets_stiminfo", "deeprecon_artificial_list.tsv"), sep="\t")
    nsd_artificial = pd.read_csv(os.path.join(save_root, "stimuli", "datasets_stiminfo", "nsd_artificial_list.tsv"), sep="\t")

    artificial_list = deeprecon_artificial.iloc[:, 0].tolist() + nsd_artificial.iloc[:, 0].tolist() #can add more to this with more datasets
    assert(len(artificial_list) == 50 + 284) 

    datasets = ['NSD','BMD','BOLD5000','THINGS','GOD','deeprecon','HAD','NOD']
    short_long_dataset_mapping = {'NSD': 'NaturalScenesDataset',
                                "BMD": "BOLDMomentsDataset",
                                "BOLD5000":"BOLD5000",
                                "THINGS": "THINGS_fmri",
                                "GOD": "GenericObjectDecoding",
                                "deeprecon": "deeprecon",
                                "HAD": "HumanActionsDataset",
                                "NOD": "NaturalObjectDataset"}

    #step 01: initialize a list of dictionaries that will be the test and train split. This naively puts the stimuli into test/train,
    #ignoring the fact there may be conflicts between the splits. that gets dealt with next.
    print("Starting step 01: initializing test and train sets according the compiled tsv. This step does not take into account conflicts.")
    train = {}
    test = {}
    for idx, filename in tqdm(enumerate(compiled_filenames), total=len(compiled_filenames), desc="Step 01: initializing test/train jsons"):
        filename_train = {}
        filename_test = {}
        for dataset in datasets:
            repetition_columns = [col for col in compiled_stiminfo.columns if f'_reps_{dataset}' in col] 
            test_or_train = compiled_stiminfo.loc[idx, f'test_train_{dataset}'] #these datasets have already been checked to ensure that each stimulus is either test or train, no overlap
            if test_or_train not in ['test', 'train']:
                continue
            else: #this path means that subject(s) from this dataset saw this stimuli. Now we put it into the correct test or train
                seen_by = {}
                for col in repetition_columns: #loop over the sub-XX_reps_{dataset} columns for one dataset
                    sub_dataset = col.replace('_reps', '')
                    numreps = compiled_stiminfo.loc[idx, col]
                    if numreps > 0:
                        fmri_rep_names = [f'{sub_dataset}_stimulus-{Path(filename).stem}_phase-{test_or_train}_rep-{rep}.npz' for rep in range(numreps)]
                        seen_by.setdefault(sub_dataset, []).extend(fmri_rep_names)
            if test_or_train == 'train':
                filename_train[dataset] = seen_by
            elif test_or_train == 'test':
                filename_test[dataset] = seen_by

        #this makes no assumptions that a stimulus is only train or only test.
        if filename_train: 
            assert filename not in train.keys(), f"filename {filename} should not be duplicated within a train/test list"    
            train[filename] = filename_train
        if filename_test:
            assert filename not in test.keys(), f"filename {filename} should not be duplicated within a train/test list"    
            test[filename] = filename_test
    #step 02: identify stimuli that are 'train' set in at least one dataset and 'test' set in at least another. whichever set has the most combined
    #repetitions wins that stimulus. For example, if a repeated image is marked 'train' in NSD and has a combined 3 repeats over all NSD subjects,
    #  and marked 'test' in BOLD5000 with a combined 4 repeats over all BOLD5000 subjects, then the stimulus will be 'test' and the NSD instances of
    #that stimulus are not included in the final split.
    print("Starting step 02: of the stimuli that are in both test and train sets, remove the one that have fewer repetitions.")
    lost_train = 0
    lost_test = 0
    duplicates = []
    for idx, filename in tqdm(enumerate(compiled_filenames), total=len(compiled_filenames), desc="Step 02: remove test/train conflicts"):
        if (filename in train) and (filename in test):
            tmp = {"train_instance": {}, "test_instance": {}, "winner": ""} #log the conflict and who wins
            #this is a test/train conflict. whoever has more repetitions wins.
            train_count = 0
            for dset in train[filename].keys():
                for sub in train[filename][dset].keys():
                    train_count += len(train[filename][dset][sub])
            
            test_count = 0
            for dset in test[filename].keys():
                for sub in test[filename][dset].keys():
                    test_count += len(test[filename][dset][sub])

            tmp['train_instance'] = train[filename]
            tmp['test_instance'] = test[filename]
            
            if train_count <= test_count:#tie goes to test set
                lost_train += train_count
                del train[filename]
                tmp['winner'] = 'test'
            elif test_count < train_count: 
                lost_test += test_count
                del test[filename]
                tmp['winner'] = 'train'
            else:
                raise ValueError(f"key {filename} not found but it should be there. Error in json formatting.")
            duplicates.append(tmp)

    print(f"Resolved {len(duplicates)} filename conflicts between test and train sets.")
    print(f"{sum([1 for dup in duplicates if dup['winner'] == 'test'])} conflicts went to test, and {sum([1 for dup in duplicates if dup['winner'] == 'train'])} to train")
    print(f"Training set lost {lost_train} repetitions.")
    print(f"Testing set lost {lost_test} repetitions.")

    #step 03: identify highly similar/identical stimuli between the test set and train set. Remove the train set reps to mitigate overfitting/classification of these instances
    print("Starting step 03: Remove train stimuli that are perceptually close to the test stimuli.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    train_filenames = list(train.keys())
    data_similarity = {img: [] for img in test.keys()} #holds the top X most similar train images to each test image.
    if not os.path.exists(os.path.join(save_root, f"train_embeddings_list.pkl")):
        print("preloaded training embeddings not found. Loading them individually now...")
        # Load all train embeddings into a batch tensor
        train_embeddings = []
        for train_img in tqdm(train_filenames, total=len(train_filenames), desc="loading individual training set embeddings"):
            #dset = short_long_dataset_mapping[list(train[train_img].keys())[0]]
            embedding_path = glob.glob(os.path.join(save_root, "model_features", "dreamsim_feats", f"{Path(train_img).stem}_model-dreamsim.npy")) #glob.glob(os.path.join(args.dataset_root, dset, "derivatives", "stimuli_metadata", "dreamsim_embeddings", f"{Path(train_img).stem}_*.npy"))
            assert len(embedding_path) == 1, f'image {train_img} had multiple dreamsim feature files associated with it. Should only be one.'
            train_img_embedding = np.load(embedding_path[0])
            train_embeddings.append(train_img_embedding)
        #save the pre-loaded train embeddings for faster similarity measures later
        with open(os.path.join(save_root, f"train_embeddings_list.pkl"), 'wb') as f:
            pickle.dump(train_embeddings, f)
    else:
        print("loading preloaded training embeddings...")
        with open(os.path.join(save_root, f"train_embeddings_list.pkl"), 'rb') as f:
            train_embeddings = pickle.load(f)
    train_embeddings = torch.tensor(np.array(train_embeddings), dtype=torch.float32, device=device)

    # Iterate over test images and calculate similarity with all train embeddings at once
    for test_img in tqdm(test.keys(), total=len(test.keys()), desc="computing similarity between testing images and training images..."):
        #dset = short_long_dataset_mapping[list(test[test_img].keys())[0]]
        embedding_path = glob.glob(os.path.join(save_root, "model_features", "dreamsim_feats", f"{Path(test_img).stem}_model-dreamsim.npy"))  #glob.glob(os.path.join(args.dataset_root, dset, "derivatives", "stimuli_metadata", "dreamsim_embeddings", f"{Path(test_img).stem}_*.npy"))
        assert len(embedding_path) == 1, f'embedding path {embedding_path} not valid'

        test_img_embedding = np.load(embedding_path[0])
        test_img_embedding = torch.tensor(test_img_embedding, dtype=torch.float32, device=device).unsqueeze(0)

        # Calculate cosine similarity between test embedding and all train embeddings in one batch operation
        similarity = cosine_similarity(test_img_embedding, train_embeddings).cpu().numpy()  # Convert result to CPU if needed
        sorted_similarity_indices = np.argsort(similarity)[::-1] #sort high similarity to low similarity
        similarity_sorted = [similarity[i] for i in sorted_similarity_indices]
        
        outlier_cutoff = 0.8196
        outlier_filenames = []
        outlier_scores = []
        outlier_indices = []
        for score, idx in zip(similarity_sorted, sorted_similarity_indices):
            if score > outlier_cutoff:
                outlier_filenames.append(train_filenames[idx])
                outlier_scores.append(score)
                outlier_indices.append(idx)
        data_similarity[test_img]=(outlier_filenames, outlier_indices, outlier_scores, outlier_cutoff) 

    #use this file later to visualize similar images and exclude
    with open(os.path.join(save_root, f"top_similar_framemetric.pkl"), 'wb') as f:
        pickle.dump(data_similarity, f)

    #If the similarity exceeds a threshold, remove the train image(s) from the train json
    removed_filenames = {}
    numreps_lost = 0 #all lost reps here are from the training set
    for test_stim, outlier_tuple in data_similarity.items():
        outlier_filenames, outlier_indices, outlier_scores, outlier_cutoff = outlier_tuple #if a similarity exceeds this threshold, discard it. This threshold is empirically determined by looking at the score between two frames of the same video but offset, which is ~0.88
        high_sim_train = [] #multiple test images can have the same train stim that are too perceptuallly 'close'. this variable tracks which train stim actually got removed because of each test stim
        for filename in outlier_filenames:
            if filename in train.keys(): #the filename could have been previously deleted and not exist anymore
                for dset in train[filename].keys():
                    for sub in train[filename][dset].keys():
                        numreps_lost += len(train[filename][dset][sub])
                del train[filename]
                high_sim_train.append(filename)
        if len(high_sim_train) > 0:
            assert test_stim not in removed_filenames.keys(), f"test stimulus {test_stim} already accounted for."
            removed_filenames[test_stim] = high_sim_train

    count = 0
    for _, v in removed_filenames.items():
        count += len(v)
    print(f"{count} stimuli were removed because of high similarity to one of {len(removed_filenames)} test stimuli.")
    print(f"A total of {numreps_lost} training repetitions were removed from the similarity filter.")

    with open(os.path.join(save_root, 'perceptually_similar_exclusions.pkl'), 'wb') as f:
        pickle.dump(removed_filenames, f)

    #reorganize the train and test sets so each index is "sub-XX_DATASET_stimulus-STIMULUSNAME" and values are responses
    #splitting up the test set into naturalistic and artificial here allows all stimuli to be involved in the filename test/train conflicts
    # and similarity filtering.
    train_naturalistic = []
    for filename in train.keys():
        filename_dict = train[filename]
        for dataset in filename_dict.keys():
            dataset_dict = filename_dict[dataset]
            for subject in dataset_dict.keys():
                responses = dataset_dict[subject]
                index_name = f"{subject}_stimulus-{filename}"
                train_naturalistic.append({index_name: responses})

    test_naturalistic = []
    for filename in test.keys():
        if filename in artificial_list:
            continue
        filename_dict = test[filename]
        for dataset in filename_dict.keys():
            dataset_dict = filename_dict[dataset]
            for subject in dataset_dict.keys():
                responses = dataset_dict[subject]
                index_name = f"{subject}_stimulus-{filename}"
                test_naturalistic.append({index_name: responses})

    test_artificial = []
    for filename in artificial_list:
        filename_dict = test[filename]
        for dataset in filename_dict.keys():
            dataset_dict = filename_dict[dataset]
            for subject in dataset_dict.keys():
                responses = dataset_dict[subject]
                index_name = f"{subject}_stimulus-{filename}"
                test_artificial.append({index_name: responses})

    print("saving test and train json files...")
    with open(os.path.join(save_root, 'train_naturalistic.json'), 'w') as f:
        json.dump(train_naturalistic, f)
    with open(os.path.join(save_root, 'test_naturalistic.json'), 'w') as f:
        json.dump(test_naturalistic, f)
    with open(os.path.join(save_root, 'test_artificial.json'), 'w') as f:
        json.dump(test_artificial, f)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets")) #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)