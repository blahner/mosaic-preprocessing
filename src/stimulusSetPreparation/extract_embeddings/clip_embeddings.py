import os
from dotenv import load_dotenv
load_dotenv()
import av
import torch
from tqdm import tqdm
import numpy as np
import argparse
import glob
import pandas as pd
from pathlib import Path
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download
import warnings
from pycocotools.coco import COCO
import h5py
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(0)

def main(args):
    #housekeeping
    if args.dataset == 'NaturalScenesDataset':
        nsd_root = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata")
        save_root = os.path.join(nsd_root, f"{args.model_name.replace('/','_')}_embeddings")
        stimuli_path = os.path.join(nsd_root,"nsd_stimuli.hdf5")
        print("loading nsd stimuli...")
        with h5py.File(stimuli_path, 'r') as hdf:
            # Assume that images are stored under a key, e.g., 'images'
            filepaths = hdf['imgBrick'][:]  # Load all images, rougly 39 gb in size. call the variable filepaths for compatibility with other datasets
        assert(filepaths.shape[0] == 73000)
        
        #load info to get nsd filenames
        nsd_csv = pd.read_csv(os.path.join(nsd_root, "nsd_stim_info_merged.csv"))
        coco_annotation_val = COCO(annotation_file=os.path.join(nsd_root, "annotations_trainval2017", "annotations", "instances_val2017.json"))
        coco_annotation_train = COCO(annotation_file=os.path.join(nsd_root, "annotations_trainval2017", "annotations", "instances_train2017.json"))
        img_ids_val = coco_annotation_val.getImgIds()
        img_ids_train = coco_annotation_train.getImgIds()
        notshown = pd.read_csv(os.path.join(nsd_root, 'notshown.tsv'), header=None).values.flatten().tolist() #nsdIDs 1-indexed
    elif args.dataset == 'BOLD5000':
        save_root = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", f"{args.model_name.replace('/','_')}_embeddings")
        stimuli_path_coco = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "Scene_Stimuli", "Presented_Stimuli", "COCO", "*.jpg")
        stimuli_path_imagenet = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "Scene_Stimuli", "Presented_Stimuli", "ImageNet", "*.JPEG")
        stimuli_path_scene_jpg = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "Scene_Stimuli", "Presented_Stimuli", "Scene", "*.jpg")
        stimuli_path_scene_jpeg = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "Scene_Stimuli", "Presented_Stimuli", "Scene", "*.jpeg")

        #get all filepaths of the stimuli
        filepaths = glob.glob(stimuli_path_coco) + glob.glob(stimuli_path_imagenet) + glob.glob(stimuli_path_scene_jpg) + glob.glob(stimuli_path_scene_jpeg)
        assert(len(filepaths) == 4916)
    elif args.dataset == 'GenericObjectDecoding':
        save_root = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", f"{args.model_name.replace('/','_')}_embeddings")
        stimuli_path_test = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "images", "test", "*.JPEG")
        stimuli_path_train = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "images", "training", "*.JPEG")

        #get all filepaths of the stimuli
        filepaths = glob.glob(stimuli_path_test) + glob.glob(stimuli_path_train)
        assert(len(filepaths) == 1250)
    elif args.dataset == 'NaturalObjectDataset':
        save_root = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", f"{args.model_name.replace('/','_')}_embeddings")
        stimuli_folders_imagenet = glob.glob(os.path.join(args.dataset_root, args.dataset, "Nifti", "stimuli", "imagenet","*"))
        #get all filepaths of the stimuli
        filepaths = []
        for folder in stimuli_folders_imagenet:
            filepaths.extend(glob.glob(os.path.join(folder, '*.JPEG')) + glob.glob(os.path.join(folder, '*.jpg')))
        
        stimuli_path_coco = os.path.join(args.dataset_root, args.dataset, "Nifti", "stimuli", "coco","*.jpg")
        filepaths.extend(glob.glob(stimuli_path_coco))
        assert(len(filepaths) == 57120)
    elif args.dataset == 'deeprecon':
        save_root = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", f"{args.model_name.replace('/','_')}_embeddings")
        stimuli_path_artificial_image = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "ArtificialImage", "*.tiff")
        stimuli_path_letter_image = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "LetterImage", "*.tif")
        #the task stimuli is the exact same as GOD
        stimuli_path_test = os.path.join(args.dataset_root, "GenericObjectDecoding", "derivatives", "stimuli_metadata", "images", "test", "*.JPEG")
        stimuli_path_train = os.path.join(args.dataset_root, "GenericObjectDecoding", "derivatives", "stimuli_metadata", "images", "training", "*.JPEG")

        #get all filepaths of the stimuli
        filepaths = glob.glob(stimuli_path_test) + glob.glob(stimuli_path_train) + glob.glob(stimuli_path_artificial_image) + glob.glob(stimuli_path_letter_image)

        assert(len(filepaths) == 1300) #1200 train, 50 test, 40 artificial, 10 letter
    elif args.dataset == 'THINGS_fmri':
        save_root = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", f"{args.model_name.replace('/','_')}_embeddings")
        stimuli_path = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "experimental_images","*.jpg")
        #get all filepaths of the stimuli
        filepaths = glob.glob(stimuli_path)
        assert(len(filepaths) == 8740)
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    #Use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device {device}")

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = AutoProcessor.from_pretrained(args.model_name)

    for idx, file_path in tqdm(enumerate(filepaths), total=len(filepaths), desc=f"Extracting {args.model_name} embeddings from {args.dataset} stimuli"):
        if args.dataset == 'NaturalScenesDataset':
            if idx+1 in notshown:
                continue #don't add the NSD images that were not shown. account for the 1 and 0 indexing difference
            cocoID = nsd_csv.loc[idx,'cocoId']
            #coco id to image filename
            if cocoID in img_ids_val:
                img_info = coco_annotation_val.loadImgs(int(cocoID))[0]
            else:
                img_info = coco_annotation_train.loadImgs(int(cocoID))[0]
            coco_filename = img_info['file_name']
            stimulus_name = Path(coco_filename).stem
            if os.path.exists(os.path.join(save_root, f"{stimulus_name}_model-{args.model_name.replace('/','_')}_nsd73kid-{idx:05}.npy")):
                continue
        else:
            stimulus_name = Path(file_path).stem        
            if os.path.exists(os.path.join(save_root, f"{stimulus_name}_model-{args.model_name.replace('/','_')}.npy")):
                continue
        
        if args.dataset == 'NaturalScenesDataset':
            image = Image.fromarray(file_path)
        else:
            image = Image.open(file_path)
        
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()} 

        image_features = model.get_image_features(**inputs)
        image_features_arr = image_features.cpu().detach().numpy()
        if args.dataset == 'NaturalScenesDataset':
            np.save(os.path.join(save_root, f"{stimulus_name}_model-{args.model_name.replace('/','_')}_nsd73kid-{idx:05}.npy"), image_features_arr)
        else:
            np.save(os.path.join(save_root, f"{stimulus_name}_model-{args.model_name.replace('/','_')}.npy"), image_features_arr)


if __name__=='__main__':
    dataset_root_default = os.getenv("DATASETS_ROOT", "/default/path/to/datasets") #use default if DATASETS_ROOT env variable is not set.

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="The fmri dataset you want to extract embeddings from.")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14", help="the model you want to use to extract embeddings.")
    parser.add_argument("--dataset_root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    args = parser.parse_args()
    
    main(args)