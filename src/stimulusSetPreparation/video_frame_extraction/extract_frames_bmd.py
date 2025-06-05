from dotenv import load_dotenv
load_dotenv()
import os
import cv2
from tqdm import tqdm
from pathlib import Path

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    success, image = vidcap.read()
    count = 1

    # Iterate through the video frames
    while success:
        # Write the current frame to a file
        cv2.imwrite(os.path.join(output_folder, f"frame-{count:04}_{total_frames:04}.jpg"), image)
        success, image = vidcap.read()
        count += 1

    vidcap.release()

#extract and save frames from a mp4 video
dataset_root = os.path.join(os.getenv('DATASETS_ROOT'),"BOLDMomentsDataset")
stim_root = os.path.join(dataset_root, "derivatives", "stimuli_metadata", "mp4_h264")
save_root = os.path.join(dataset_root, "derivatives", "stimuli_metadata", "frames")
if not os.path.exists(save_root):
    os.makedirs(save_root)

videos = os.listdir(stim_root)
assert(len(videos) == 1102)

print("extracting frames...")
for vid in tqdm(videos, total=len(videos)):
    video_path = os.path.join(stim_root, vid)
    video_noext = Path(vid).stem
    output_folder = os.path.join(save_root, video_noext)
    extract_frames(video_path, output_folder)