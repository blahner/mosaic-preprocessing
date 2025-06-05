from dotenv import load_dotenv
load_dotenv()
import numpy as np
import os
from tqdm import tqdm
import glob
from moviepy.editor import VideoFileClip

datasets_root = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"))

#the "stimuli_metadata" folder is reorganized to contain the original 23 .mp4 files in a folder called 'original' and this is where we write derivatives of these stimuli, such as the 2s clipped verison.
stim_root = os.path.join(datasets_root, "CC2017", "video_fmri_dataset", "stimuli_metadata")
if not os.path.exists(stim_root):
    os.makedirs(stim_root)
clip_length = 2 #in seconds

save_root = os.path.join(stim_root, f"clipped_{clip_length}s", "mp4")
if not os.path.exists(save_root):
    os.makedirs(save_root)

full_paths =  glob.glob(os.path.join(stim_root, "original", "*.mp4"))
assert len(full_paths) == 23, f"Expected 23 videos, but found {len(full_paths)}." 

for clip_path in tqdm(full_paths, desc="Processing Clips"):  # Added tqdm for progress tracking
    head, tail = os.path.split(clip_path)
    fname = tail.split('.mp4')[0]
    clip = VideoFileClip(clip_path)  # load clip to get its duration for accurate segmenting

    # Use np.isclose() for float comparison with a tolerance
    assert np.isclose(np.floor(clip.duration), 480), f"Clip duration is not 480 seconds, but {clip.duration}."

    num_clips = int(np.floor(clip.duration) / clip_length)
    t1 = 0  # t1 is always 0
    for i in range(num_clips):
        t2 = t1 + clip_length
        subclip = clip.subclip(t1, t2)
        subclip.write_videofile(os.path.join(save_root, f"{fname}_begin-{t1:03}_end-{t2:03}.mp4"),
                                codec='libx264', audio_codec='aac')
        t1 = t2
