from dotenv import load_dotenv
load_dotenv()
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from PIL import Image
import torch
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import glob
from torchvision.transforms import v2
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_tsfm = v2.Compose([v2.Resize((224, 224)),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                    )

    # Load pretrained weights
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model = model.to(device)
    train_nodes, eval_nodes = get_graph_node_names(model) #list of model layers or nodes
    model.eval()
    # get activations from all samples in the dataloader
    interested_layer = 'features.10' #layers I want the activations from. Must be in "nodes"
    assert interested_layer in eval_nodes, f"Error: {interested_layer} not found in AlexNet model layers."
    embeddings_root = os.path.join(args.dataset_root, "model_features", f"alexnetPytorch_{interested_layer.replace('.','_')}_feats")
    if not os.path.exists(embeddings_root):
        os.makedirs(embeddings_root)

    #define feature extractor for your desired layer
    feature_extractor = create_feature_extractor(model, return_nodes=[interested_layer]) #the term "feature" and "activation" is interchangeable here

    #get all stimulus paths for video frames and images
    print("Gathering stimulus paths...")
    stimulus_path = os.path.join(args.dataset_root, 'stimuli')
    video_dict = {}
    video_paths = glob.glob(os.path.join(stimulus_path, "raw", f"*.mp4"))
    for video_path in tqdm(video_paths, total=len(video_paths), desc="compiling video frames"):
        video_name = Path(video_path).stem
        frames = sorted(glob.glob(os.path.join(stimulus_path, "frames", video_name, "*.jpg")))
        video_dict[video_name] = frames[1:-1] #dont take the first or last frame to handle edge effects
    print(f"Found {len(video_dict)} videos.")

    img_list = []
    for ext in ['.jpg', '.JPG', '.JPEG', '.jpeg']:
        img_list.extend(glob.glob(os.path.join(stimulus_path, "raw", f"*{ext}")))
        img_list.extend(glob.glob(os.path.join(stimulus_path, "tiff2jpg", f"*{ext}"))) #these were tiff/tif images converted to jpg. net2brain cannot take tiff as input
    img_list.sort()
    print(f"Found {len(img_list)} images.")

    #video frame features are averaged together
    for video_filename, frames in tqdm(video_dict.items(), total=len(video_dict), desc="Extracting AlexNet embeddings for video stimuli"):
        if os.path.isfile(os.path.join(embeddings_root, f"{video_filename}_model-alexnetPytorch_{interested_layer.replace('.','_')}.npy")):
            continue
        embedding = 0
        for frame in frames:
            with Image.open(frame).convert("RGB") as img:
                stim = img.copy() 
            img = img_tsfm(stim).to(device)
            ft = feature_extractor(img)
            embedding += ft[interested_layer].cpu().detach().numpy().flatten()
        embedding = embedding / len(frames) #average
        np.save(os.path.join(embeddings_root, f"{video_filename}_model-alexnetPytorch_{interested_layer.replace('.','_')}.npy"), embedding)  

    for img_filename in tqdm(img_list, total=len(img_list), desc="Extracting AlexNet embeddings for image stimuli"):
        if os.path.isfile(os.path.join(embeddings_root, f"{Path(img_filename).stem}_model-alexnetPytorch_{interested_layer.replace('.','_')}.npy")):
            continue
        with Image.open(img_filename).convert("RGB") as img:
            stim = img.copy()         
        img = img_tsfm(stim).to(device)
        ft = feature_extractor(img)
        embedding = ft[interested_layer].cpu().detach().numpy().flatten()
        np.save(os.path.join(embeddings_root, f"{Path(img_filename).stem}_model-alexnetPytorch_{interested_layer.replace('.','_')}.npy"), embedding)

if __name__=='__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC") #use default if DATASETS_ROOT env variable is not set.
    tmp_root_default = os.path.join(os.getenv("CACHE", "/path/to/tmp/directory")) 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    args = parser.parse_args()
    
    main(args)