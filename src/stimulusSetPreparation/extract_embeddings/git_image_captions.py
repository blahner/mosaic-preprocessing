from dotenv import load_dotenv
load_dotenv()
import os
import glob as glob
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import json

datasets = ['deeprecon',
            'NaturalScenesDataset',
            'BOLD5000',
            'GenericObjectDecoding',
            'THINGS_fmri',
            'NaturalObjectDataset']
device = "cuda" if torch.cuda.is_available() else "cpu"
#load model
model_flavor = "git-large-coco" #"microsoft/git-large-textcaps" #"microsoft/git-base-coco"
processor = AutoProcessor.from_pretrained(f"microsoft/{model_flavor}")
model = AutoModelForCausalLM.from_pretrained(f"microsoft/{model_flavor}").to(device)

for dataset in datasets:
    dataset_root = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),dataset) #use default if DATASETS_ROOT env variable is not set.

    save_root = os.path.join(dataset_root, "derivatives", "stimuli_metadata")

    if dataset == 'BOLDMomentsDataset':
        stimuli = [f"{vid:04}" for vid in range(1,1103)]
        annotations = {video: {f"GIT-{model_flavor}": []} for video in stimuli}

    for stim in stimuli:
        frame_path = os.path.join(dataset_root, "derivatives", "stimuli_metadata","frames",stim)
        total_frames = len(glob.glob(os.path.join(frame_path, f"{stim}_*.jpg")))
        middle_frame = Image.open(os.path.join(frame_path, f"{stim}_{int(total_frames/2)}_{total_frames}.jpg"))
        pixel_values = processor(images=middle_frame, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values=pixel_values, max_length=100, num_return_sequences=5, num_beams=5, temperature=1.0, top_k=250, top_p=1.0)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)

        print(f"{stim}: {generated_caption}")
        annotations[stim][f"GIT-{model_flavor}"] = generated_caption

    with open(os.path.join(save_root, f"GIT_captions.json"), 'w') as f:
        json.dump(annotations, f)
