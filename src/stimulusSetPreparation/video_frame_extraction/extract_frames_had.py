from dotenv import load_dotenv
load_dotenv()
import os
from tqdm import tqdm
import cv2

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
dataset_root = os.path.join(os.getenv('DATASETS_ROOT'),"HumanActionsDataset")
stim_root = os.path.join(dataset_root, "Nifti", "stimuli")
save_root = os.path.join(dataset_root, "derivatives", "stimuli_metadata", "frames")
if not os.path.exists(save_root):
    os.makedirs(save_root)

cat_dirs = os.listdir(stim_root)
assert(len(cat_dirs) == 180)

print("extracting frames...")
for cat in tqdm(cat_dirs, total=len(cat_dirs)):
    print(f"category: {cat}")

    vid_dirs = os.listdir(os.path.join(stim_root, cat))
    assert(len(vid_dirs) == 120)
    for vid in vid_dirs:
        video_path = os.path.join(stim_root, cat, vid)
        video_noext = vid.split('.mp4')[0]
        output_folder = os.path.join(save_root, cat, video_noext)
        extract_frames(video_path, output_folder)