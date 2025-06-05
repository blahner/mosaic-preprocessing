from dotenv import load_dotenv
load_dotenv()
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path
import glob
import argparse


"""
Aggressively downsample the image and middle frame stimuli
so it can be used for visualization in webpages.
"""
def compress_image(input_path, output_path, quality=30, new_size=None):
    # Open the image
    with Image.open(input_path) as img:
        # If new size is provided, resize the image
        if new_size:
            img = img.resize(new_size, Image.LANCZOS)
        
        # Save the image with reduced quality
        img.save(output_path, "JPEG", quality=quality, optimize=True)

def main(args):
    stim_root = os.path.join(args.dataset_root, "MOSAIC", "stimuli")
    raw_dir = os.path.join(stim_root, "raw")
    middleframes_dir = os.path.join(stim_root, "frames_middle")
    save_root = os.path.join(stim_root, f"stimuli_compressed_quality-{args.quality}_size-{args.size}")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    #collect all stimuli filepaths.
    images_all = glob.glob(os.path.join(raw_dir, '*.jpg')) + \
        glob.glob(os.path.join(raw_dir, '*.JPG')) + \
        glob.glob(os.path.join(raw_dir, '*.JPEG')) + \
        glob.glob(os.path.join(raw_dir, '*.jpeg')) + \
        glob.glob(os.path.join(raw_dir, '*.tif')) + \
        glob.glob(os.path.join(raw_dir, '*.tiff')) + \
        glob.glob(os.path.join(middleframes_dir, '*.jpg'))

    for input_path in tqdm(images_all, total=len(images_all), desc=f"resizing and compressing stimuli..."):
        filename = Path(input_path).name

        #Resize and compress
        compress_image(input_path, os.path.join(save_root, filename), quality=args.quality, new_size=(args.size, args.size))

if __name__=='__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets")) #use default if DATASETS_ROOT env variable is not set.
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    parser.add_argument("--quality", type=int, required=True, help="The quality of compression from 0 to 95. Lower the value the more aggressive compression.")
    parser.add_argument("--size", type=int, default=dataset_root_default, help="size in pixels to resize the image to. square resize is default.")

    args = parser.parse_args()
    
    main(args)