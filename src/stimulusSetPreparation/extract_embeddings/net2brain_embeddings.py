from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import argparse

#local
from src.stimulusSetPreparation.extract_embeddings.extractor_functions import extract_net2brain_model_features

np.random.seed(0)

def main(args):
    #by default all images from all datasets are being extracted. If a dataset uses videos (BMD, HAD) then the middle frame image is used.
    extract_net2brain_model_features(args.model_name, save_root=args.dataset_root, dummy_run=False)

if __name__=='__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC") #use default if DATASETS_ROOT env variable is not set.

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="AlexNet", help="the model you want to use to extract embeddings.")
    parser.add_argument("--dataset_root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    args = parser.parse_args()
    
    main(args)