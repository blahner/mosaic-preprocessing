from dotenv import load_dotenv
load_dotenv()
import cv2
import os
import glob as glob
from tqdm import tqdm 
 
def extract_frames(root, video_path, output_folder_all, output_folder_middle):
    video_filename = os.path.basename(video_path)
    filename_no_ext = os.path.splitext(video_filename)[0]
    output_dir_all = os.path.join(root, output_folder_all, filename_no_ext)
    output_dir_middle = os.path.join(root, output_folder_middle)
    # Create directories if they don't exist
    os.makedirs(output_dir_all, exist_ok=True)
    os.makedirs(output_dir_middle, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        raise ValueError
        #return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the middle frame index
    middle_frame_index = total_frames // 2
    
    # Loop through all frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save all frames in the output directory for all frames
        frame_filename = os.path.join(output_dir_all, f"frame_{frame_count:03d}_{total_frames}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Save the middle frame in a separate directory
        if frame_count == middle_frame_index:
            middle_frame_filename = os.path.join(output_dir_middle, f"{filename_no_ext}_frame_{frame_count:03d}_{total_frames:03d}.jpg")
            cv2.imwrite(middle_frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"All frames saved in: {output_dir_all}")
    print(f"Middle frame saved in: {output_dir_middle}")
    
# Example usage
datasets_root = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"))
root = os.path.join(datasets_root, "CC2017", "video_fmri_data", "stimuli_metadata", "clipped_2s") #run 'cip_stimuli.py' first to extract 2s clips into this folder.
output_folder_all_frames = "frames"
output_folder_middle_frame = "frames_middle"
all_mp4s = glob.glob(os.path.join(root, 'mp4', '*.mp4'))
for video_path in tqdm(all_mp4s, total=len(all_mp4s), desc="extracting frames from CC2017 2s snippets..."):
    extract_frames(root, video_path, output_folder_all_frames, output_folder_middle_frame)
