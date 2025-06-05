from dotenv import load_dotenv
load_dotenv()
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import argparse
from tqdm import tqdm

def main(args):
    model_name = 'paraphrase-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    save_root = os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", f"imagenet_category_embeddings_{model_name}")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    #imagenet
    with open(os.path.join(args.dataset_root,"derivatives", "stimuli_metadata", "testtrain_split", "synset_words_edited.txt"), 'r') as f:
        # Initialize lists to store the columns
        imagenet_names = []
        category_labels = []

        # Iterate through each line in the file
        for line in f:
            # Split the line at the first space to get the 'n*' code and the labels
            parts = line.strip().split(' ', 1)  # Split on first space only
            imagenet_names.append(parts[0])  # First part is the imagenet name
            category_labels.append(parts[1])  # The rest is the label

        # Create a DataFrame with custom column names
        metadata = pd.DataFrame({
            'imagenet_name': imagenet_names,
            'category_labels': category_labels
        })

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="extracting embeddings..."):
        imagenet_name = row['imagenet_name']
        labels = row['category_labels'].split(',') #multiple labels (e.g., synonyms of the same label) are separated by commas. average the embeddings
        embedding = np.zeros((384,))
        clean_labels = []
        for label in labels:
            clean_labels.append(label.strip().replace("'","").lower())
        for label in clean_labels:
            embedding += model.encode(label)
        embedding = embedding/len(clean_labels) #average

        #save embedding
        np.save(os.path.join(save_root, f"imagenetName-{imagenet_name}_category-{clean_labels[0].replace(' ','_')}_model-{model_name}_embedding.npy"), embedding)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"NaturalObjectDataset") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)