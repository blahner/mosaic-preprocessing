from dotenv import load_dotenv
load_dotenv()
import os
import av
import numpy as np
import json
import glob
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

datasets = ['HumanActionsDataset'] #['BOLDMomentsDataset','HumanActionsDataset']
device = "cuda" if torch.cuda.is_available() else "cpu"

model_flavor = "git-large-vatex"
processor = AutoProcessor.from_pretrained(f"microsoft/{model_flavor}")
model = AutoModelForCausalLM.from_pretrained(f"microsoft/{model_flavor}").to(device)

# set seed for reproducability
np.random.seed(45)

for dataset in datasets:
    dataset_root = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),dataset) #use default if DATASETS_ROOT env variable is not set.

    save_root = os.path.join(dataset_root, "derivatives", "stimuli_metadata")
    if dataset == 'BOLDMomentsDataset':
        video_root = os.path.join(dataset_root, "derivatives", "stimuli_metadata", "mp4_h264")
        file_paths = glob.glob(os.path.join(dataset_root, "derivatives", "stimuli_metadata","mp4_h264","*.mp4"))
        stimuli = [f"{vid:04}.mp4" for vid in range(1,1103)]
        annotations = {video: {f"GIT-{model_flavor}": []} for video in stimuli}
    elif dataset == 'HumanActionsDataset':
        video_root = os.path.join(dataset_root, "Nifti", "stimuli")
        frame_paths = glob.glob(os.path.join(dataset_root, "derivatives", "stimuli_metadata","frames_middle","*.jpg")) #just to get the filenames
        stimuli = []
        for frame_path in frame_paths:
            filename = Path(frame_path).stem.split('_frame')[0]
            action_folder_tmp = filename.split('_id_')[0]
            action_folder = action_folder_tmp.split('v_')[-1]
            stimuli.append(f"{action_folder}/{filename}.mp4")
        annotations = {video: {f"GIT-{model_flavor}": []} for video in stimuli}

    for stim in tqdm(stimuli):
        file_path = os.path.join(video_root, stim)
        # load video
        container = av.open(file_path)

        # sample frames
        num_frames_to_sample = model.config.num_image_with_embedding
        total_frames = container.streams.video[0].frames
        indices = np.round(np.linspace(1, total_frames-1, num_frames_to_sample)).astype(np.int64)

        frames = read_video_pyav(container, indices)

        pixel_values = processor(images=list(frames), return_tensors="pt").pixel_values.to(device)

        generated_ids = model.generate(pixel_values=pixel_values, max_length=100)
        generated_caption =  processor.batch_decode(generated_ids, skip_special_tokens=True)

        print(f"{stim}: {generated_caption}")
        annotations[stim][f"GIT-{model_flavor}"] = generated_caption

    with open(os.path.join(save_root, f"GIT_captions.json"), 'w') as f:
        json.dump(annotations, f)