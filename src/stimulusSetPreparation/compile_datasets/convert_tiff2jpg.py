from dotenv import load_dotenv
load_dotenv()
import os
import glob
import argparse
from pathlib import Path
from PIL import Image

"""
Some models (e.g., DreamSim via net2brain) do not accept .tif/.tiff inputs. This script saves a .jpg copy
of the deeprecon artificial shape (.tiff) and letter (.tif) stimuli into MOSAIC/stimuli/tiff2jpg/, keeping
the original filename stem (e.g., colorExpStim01_red_square.tiff -> colorExpStim01_red_square.jpg) so that
embeddings saved by stem still match the stimulus filenames in the stiminfo .tsv files.
The original .tif/.tiff files are still copied into MOSAIC/stimuli/raw/ by copy_dataset_stim.py.
"""

def main(args):
    source_root = os.path.join(args.dataset_root, "deeprecon", "derivatives", "stimuli_metadata", "images")
    save_root = os.path.join(args.dataset_root, "MOSAIC", "stimuli", "tiff2jpg")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    filepaths = sorted(glob.glob(os.path.join(source_root, "ArtificialImage", "*.tiff")) + glob.glob(os.path.join(source_root, "LetterImage", "*.tif")))
    assert(len(filepaths) == 50) #40 artificial shapes, 10 letters
    for source_file in filepaths:
        destination_file = os.path.join(save_root, f"{Path(source_file).stem}.jpg")
        with Image.open(source_file) as image:
            image.convert("RGB").save(destination_file, "JPEG", quality=95)
    print(f"saved {len(filepaths)} jpg stimuli to {save_root}")

if __name__ == '__main__':
    dataset_root_default = os.getenv("DATASETS_ROOT", "/default/path/to/datasets") #use default if DATASETS_ROOT env variable is not set.

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    args = parser.parse_args()

    main(args)
