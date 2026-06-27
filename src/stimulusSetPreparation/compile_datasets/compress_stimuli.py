"""
Resize and compress MOSAIC stimuli for web visualization.

All outputs are saved as  {raw_stimulus_stem}.jpg  so filenames align with:
  - embedding files  : {stem}_model-*.npy  (clip_feats_viz/, dreamsim_feats_viz/)
  - JSON split files : Path(stim_id).stem == stem
  - S3 thumbnail URL : stimuli_compressed/{stem}.jpg

Sources:
  stimuli/raw/           -- all image stimuli (.jpg .JPEG .jpeg .tiff .tif .png)
  stimuli/frames_middle/ -- one representative frame per video (.jpg)
                           saved as {mp4_stem}.jpg (frame suffix stripped)

Output:
  stimuli/stimuli_compressed_resized_quality-{quality}_size-{size}/

Usage:
    python compress_stimuli.py --quality 40 --size 112 [--workers 16]
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.tif", "*.tiff", "*.png", "*.PNG")


# ---------------------------------------------------------------------------
# Worker — must be a module-level function so it can be pickled by multiprocessing
# ---------------------------------------------------------------------------

def _process_one(task: tuple) -> str | None:
    """Compress a single image. Returns output_path on success, None if skipped."""
    input_path, output_path, quality, size = task
    if os.path.exists(output_path):
        return None
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img.thumbnail((size, size), Image.LANCZOS)
            img.save(output_path, "JPEG", quality=quality, optimize=True)
    except Exception as e:
        print(f"  Warning: failed on {input_path}: {e}")
        return None
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    stim_root      = os.path.join(args.dataset_root, "MOSAIC", "stimuli")
    raw_dir        = os.path.join(stim_root, "raw")
    frames_mid_dir = os.path.join(stim_root, "frames_middle")
    save_root      = os.path.join(stim_root,
                                  f"stimuli_compressed_resized_quality-{args.quality}_size-{args.size}")
    os.makedirs(save_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Build task list: (input_path, output_path, quality, size)
    # ------------------------------------------------------------------
    tasks = []

    # 1. Raw image stimuli → {stem}.jpg
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(glob.glob(os.path.join(raw_dir, ext)))
    for input_path in sorted(image_paths):
        stem        = Path(input_path).stem
        output_path = os.path.join(save_root, f"{stem}.jpg")
        tasks.append((input_path, output_path, args.quality, args.size))

    # 2. Video middle-frames → {mp4_stem}.jpg  (strip _frame-XXXX_YYYY suffix)
    for input_path in sorted(glob.glob(os.path.join(frames_mid_dir, "*.jpg"))):
        frame_stem  = Path(input_path).stem
        mp4_stem    = re.sub(r"_frame-\d+_\d+$", "", frame_stem)
        output_path = os.path.join(save_root, f"{mp4_stem}.jpg")
        tasks.append((input_path, output_path, args.quality, args.size))

    print(f"Total tasks : {len(tasks):,}  ({sum(os.path.exists(t[1]) for t in tasks):,} already done)")
    print(f"Workers     : {args.workers}")
    print(f"Output      : {save_root}\n")

    # ------------------------------------------------------------------
    # Run in parallel
    # ------------------------------------------------------------------
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Compressing stimuli") as pbar:
            for future in as_completed(futures):
                future.result()   # re-raises any unhandled exception
                pbar.update(1)

    print(f"\nDone. Compressed stimuli saved to:\n  {save_root}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_root_default = os.getenv("DATASETS_ROOT", "/your/path/to/datasets")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset_root", type=str, default=dataset_root_default,
                        help="Root path to datasets folder (MOSAIC/ is appended).")
    parser.add_argument("--quality", type=int, required=True,
                        help="JPEG quality 0–95. Lower = smaller files.")
    parser.add_argument("--size", type=int, default=224,
                        help="Square bounding-box size in pixels (default: 224).")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help=f"Parallel worker processes (default: all CPUs = {os.cpu_count()}).")
    args = parser.parse_args()
    main(args)
