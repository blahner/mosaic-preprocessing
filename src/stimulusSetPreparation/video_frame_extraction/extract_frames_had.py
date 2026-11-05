from dotenv import load_dotenv
load_dotenv()
import os
import glob
import shutil
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
middle_root = os.path.join(dataset_root, "derivatives", "stimuli_metadata", "frames_middle") #middle frame of each video, convenient for visualizations
for d in [save_root, middle_root]:
    if not os.path.exists(d):
        os.makedirs(d)

cat_dirs = os.listdir(stim_root)
assert(len(cat_dirs) == 180)

print("extracting frames...")
for cat in tqdm(cat_dirs, total=len(cat_dirs)):
    print(f"category: {cat}")

    vid_dirs = os.listdir(os.path.join(stim_root, cat))
    assert(len(vid_dirs) == 120)
    for vid in vid_dirs:
        video_path = os.path.join(stim_root, cat, vid)
        video_noext = vid.split('.mp4')[0] #lots of HAD videos have periods in the filename so Path(vid).stem doesnt work
        output_folder = os.path.join(save_root, cat, video_noext)
        if glob.glob(os.path.join(middle_root, f"{video_noext}_frame-*.jpg")):
            continue #already extracted (safe to re-run after an interruption)
        extract_frames(video_path, output_folder)

        #copy the middle frame, named <video>_frame-<middle>_<total>.jpg like the frames in MOSAIC/stimuli/frames
        frames = sorted(glob.glob(os.path.join(output_folder, "*.jpg")))
        assert(len(frames) > 0), f"no frames extracted from {video_path}"
        middle_frame = frames[len(frames)//2 - 1] if len(frames) > 1 else frames[0] #frame number len(frames)//2 (1-indexed)
        shutil.copyfile(middle_frame, os.path.join(middle_root, f"{video_noext}_{os.path.basename(middle_frame)}"))
