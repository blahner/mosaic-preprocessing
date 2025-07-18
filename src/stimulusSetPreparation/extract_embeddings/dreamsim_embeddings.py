from dotenv import load_dotenv
load_dotenv()
from dreamsim import dreamsim
from PIL import Image
import torch
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import glob
import re
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
This script extracts DreamSim embeddings from all stimuli in a common folder. This script
supports '.jpg', '.JPG', '.JPEG', '.jpeg' image files and .mp4 video files. Other file extensions
(e.g., .tif) may need to be converted to an acceptable file format due to what DreamSim accepts as
input. For .mp4 video files, DreamSim embeddings are averaged over frames.

You may want to extract DreamSim embeddings for stimuli in one dataset (e.g., for making test-train splits).
In that case, edit filepaths and filenames.
"""

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = dreamsim(pretrained=True, cache_dir=os.path.join(args.tmp_dir,".cache"))
    
    embeddings_root = os.path.join(args.dataset_root, "model_features", "dreamsim_feats")
    if not os.path.exists(embeddings_root):
        os.makedirs(embeddings_root)

    stimulus_path = os.path.join(args.dataset_root, 'stimuli')
    # Create lists with all training and test image file names, sorted

    print("Gathering stimulus paths...")
    video_dict = {}
    video_paths = glob.glob(os.path.join(stimulus_path, "raw", f"*.mp4"))
    for video_path in tqdm(video_paths, total=len(video_paths), desc="compiling video frames"):
        video_name = Path(video_path).stem
        frames = sorted(glob.glob(os.path.join(stimulus_path, "frames", video_name, "*.jpg")))
        video_dict[video_name] = frames[1:-1] #dont take the first or last frame to handle edge effects
    print(f"Found {len(video_dict)} videos.")

    for video_filename, frames in tqdm(video_dict.items(), total=len(video_dict), desc="Extracting DreamSim embeddings for video stimuli"):
        if os.path.isfile(os.path.join(embeddings_root, f"{video_filename}_model-dreamsim.npy")):
            continue
        embedding = 0
        for frame in frames:
            image = Image.open(frame)
            img = preprocess(image).to(device)
            embedding += model.embed(img).detach().cpu().numpy()[0]
        embedding = embedding / len(frames) #average
        np.save(os.path.join(embeddings_root, f"{video_filename}_model-dreamsim.npy"), embedding)  

    img_list = []
    for ext in ['.jpg', '.JPG', '.JPEG', '.jpeg']:
        img_list.extend(glob.glob(os.path.join(stimulus_path, "raw", f"*{ext}")))
        img_list.extend(glob.glob(os.path.join(stimulus_path, "tiff2jpg", f"*{ext}"))) #these were tiff/tif images converted to jpg. net2brain cannot take tiff as input
    img_list.sort()
    print(f"Found {len(img_list)} images.")

    for img_filename in tqdm(img_list, total=len(img_list), desc="Extracting DreamSim embeddings for image stimuli"):
        if os.path.isfile(os.path.join(embeddings_root, f"{Path(img_filename).stem}_model-dreamsim.npy")):
            continue
        print(f"extracting dreamsim embeddings for {img_filename}")
        image = Image.open(img_filename)
        img = preprocess(image).to(device)
        embedding = model.embed(img).detach().cpu().numpy()[0]
        np.save(os.path.join(embeddings_root, f"{Path(img_filename).stem}_model-dreamsim.npy"), embedding)

if __name__=='__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC") #use default if DATASETS_ROOT env variable is not set.
    tmp_root_default = os.path.join(os.getenv("CACHE", "/path/to/tmp/directory")) 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    parser.add_argument("--tmp_dir", type=str, default=tmp_root_default, help="path to a temporary directory to store dreamsim weights.")
    args = parser.parse_args()
    
    main(args)