from dotenv import load_dotenv
load_dotenv()
import os
import argparse
from tqdm import tqdm
import glob
import shutil
from pathlib import Path

"""
After all the frames are extracted from the mp4 files, loop through them and copy the middle frame of each video 
to a different folder. These middle frames are more convenient for vizualizations.
"""

def main(args):
    #setup directories
    allframes_root = os.path.join(args.dataset_root, "stimuli", "frames")
    save_root = os.path.join(args.dataset_root, "stimuli", "frames_middle")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    #loop over all video categories
    video_names_all = glob.glob(os.path.join(allframes_root, "*"))
    assert len(video_names_all) == 22702, f"Found {len(video_names_all)} but there should be 22702"
    for video in tqdm(video_names_all):
        video_name = Path(video).name
        total_frames = len(list(Path(video).glob("*.jpg")))
        middle_frame_idx = int(total_frames//2)
        middle_frame_filename = f"{video_name}_frame-{middle_frame_idx:04}_{total_frames:04}.jpg"
        shutil.copyfile(os.path.join(video, middle_frame_filename), os.path.join(save_root, middle_frame_filename))

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--dataset_root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)