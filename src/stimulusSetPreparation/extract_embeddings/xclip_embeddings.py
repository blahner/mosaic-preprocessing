import os
from dotenv import load_dotenv
load_dotenv()
import av
import torch
from tqdm import tqdm
import numpy as np
import argparse
import glob
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
images are treated as videos with duration as long as their stimulus presentation.
"""

np.random.seed(0)

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


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def main(args):
    #housekeeping
    if args.dataset == 'BOLDMomentsDataset':
        dataset_video_length=3 #in seconds
        save_root = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", f"{args.model_name.replace('/','_')}_embeddings")
        stimuli_path = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "mp4_h264", "*.mp4")
        
        #get all filepaths of the stimuli
        filepaths = glob.glob(stimuli_path)
        assert(len(filepaths) == 1102)
    elif args.dataset == 'HumanActionsDataset':
        dataset_video_length=2 #in seconds
        save_root = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", f"{args.model_name.replace('/','_')}_embeddings")
        stimuli_folders = glob.glob(os.path.join(args.dataset_root, args.dataset, "Nifti", "stimuli", "*"))
        assert(len(stimuli_folders) == 180)
        #get all filepaths of the stimuli
        filepaths = []
        for folder in stimuli_folders:
            filepaths.extend(glob.glob(os.path.join(folder, '*.mp4')))
        assert(len(filepaths) == 21600)
    elif args.dataset == 'CC2017':
        dataset_video_length=2 #in seconds
        save_root = os.path.join(args.dataset_root, args.dataset,"video_fmri_dataset", "stimuli_metadata", "clipped_2s", f"{args.model_name.replace('/','_')}_embeddings")
        stimuli_path = os.path.join(args.dataset_root, args.dataset, "video_fmri_dataset", "stimuli_metadata", "clipped_2s", "mp4", "*.mp4")
        #get all filepaths of the stimuli
        filepaths = glob.glob(stimuli_path)
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    #Use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device {device}")

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    for file_path in tqdm(filepaths, total=len(filepaths), desc=f"Extracting {args.model_name} embeddings from {args.dataset} stimuli"):
        stimulus_name = Path(file_path).stem
        if os.path.exists(os.path.join(save_root, f"{stimulus_name}_model-{args.model_name.replace('/','_')}.npy")):
            continue

        container = av.open(file_path)

        #uniformly sample X frames per second (default 4 fps) from the video.
        #don't sample the first or last frame in case of boundary effects

        # Get the video stream (assuming video stream is the first stream)
        video_stream = next(s for s in container.streams if s.type == 'video')
        duration_in_seconds = round(float(video_stream.duration * video_stream.time_base))
        assert(duration_in_seconds == dataset_video_length)

        num_frames = 8 #dataset_video_length * args.sample_fps #int(args.sample_fps*duration_in_seconds)
        total_frames = container.streams.video[0].frames
        if num_frames > total_frames:
            num_frames = total_frames - 2 #minus 2 because we do not include the first and last frame
        start_idx = 1 #0 is the first frame, 1 is the second frame
        end_idx = total_frames - 1
        indices = np.linspace(start_idx, end_idx, num=num_frames)
        indices = np.clip(indices, start_idx, end_idx).astype(np.int64)
        video = read_video_pyav(container, indices)

        inputs = processor(videos=list(video), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()} 

        video_features = model.get_video_features(**inputs)
        video_features_arr = video_features.cpu().detach().numpy()

        np.save(os.path.join(save_root, f"{stimulus_name}_model-{args.model_name.replace('/','_')}.npy"), video_features_arr)

if __name__=='__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets")) #use default if DATASETS_ROOT env variable is not set.

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="The fmri dataset you want to extract embeddings from.")
    parser.add_argument("--model_name", type=str, default="microsoft/xclip-large-patch14", help="the model you want to use to extract embeddings.")
    parser.add_argument("--dataset_root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    parser.add_argument("--sample_fps", type=int, default=4, help="How many frames per second to sample from each video.")    
    args = parser.parse_args()
    
    main(args)