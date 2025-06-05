from dotenv import load_dotenv
load_dotenv()
import os
import torch
import glob
from net2brain.feature_extraction import FeatureExtractor
from net2brain.feature_extraction import all_networks

dataset_root = os.path.join(os.getenv('DATASETS_ROOT'), "MOSAIC")

def extract_net2brain_model_features(model_name: str, save_root: str=dataset_root, dummy_run: bool=False) -> None:
    standard_netset = all_networks['Standard']
    if model_name not in standard_netset:
        raise ValueError(f"Currently only models in NetBrain's standard netset are available. \
                         You selected the model {model_name}, but the model must be one of {standard_netset}.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)

    stimulus_path = os.path.join(save_root, 'stimuli')

    # Create lists will all training and test image file names, sorted
    print("Gathering stimulus paths...")
    img_list = []
    for ext in ['.jpg', '.JPG', '.JPEG', '.jpeg']:
        img_list.extend(glob.glob(os.path.join(stimulus_path, "raw", f"*{ext}")))
        img_list.extend(glob.glob(os.path.join(stimulus_path, "frames_middle", f"*{ext}"))) #for the videos
        img_list.extend(glob.glob(os.path.join(stimulus_path, "tiff2jpg", f"*{ext}"))) #these were tiff/tif images converted to jpg. net2brain cannot take tiff as input
    img_list.sort()
    if dummy_run:
        img_list = img_list[:50]

    print(f"Found {len(img_list)} images.")

    fx_model = FeatureExtractor(model=model_name,
                                netset='Standard',
                                device=device)
    save_path = os.path.join(save_root, 'model_features', "net2brain", f'{model_name}_feats')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Extracting stimulus features...")
    #[fx_model.layers_to_extract[-1]]
    fx_model.extract(data_path=img_list,
                    save_path=save_path,
                    layers_to_extract=['features.10'],
                    consolidate_per_layer=True)