"""
Extract CLIP (openai/clip-vit-large-patch14) embeddings for all MOSAIC stimuli.

Raw stimuli:  <mosaic_root>/stimuli/raw/
  - .mp4 files : frames decoded on-the-fly with PyAV; per-frame embeddings are
                 averaged (first and last frame skipped to avoid edge effects).
  - image files: embedded directly (.jpg .JPEG .jpeg .tiff .tif .png etc.)

Embeddings saved to: <mosaic_root>/model_features/clip_feats_viz/
  Filename format  : {stimulus_stem}_model-openai_clip-vit-large-patch14.npy
  Shape            : (768,)  [float32]

Already-computed files are skipped automatically (safe to resume).

Multi-GPU: the launcher spawns one subprocess per GPU with CUDA_VISIBLE_DEVICES
set in the OS environment before Python starts, so all model internals land on
cuda:0 of each process (which maps to the correct physical GPU).

Usage:
    python clip_embeddings_viz.py                        # all available GPUs
    python clip_embeddings_viz.py --gpus 0 1             # specific GPUs
    python clip_embeddings_viz.py --gpus 4               # single GPU
"""

import os
import sys
import glob
import argparse
import subprocess
import warnings

import av
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, CLIPModel

warnings.simplefilter(action="ignore", category=FutureWarning)
np.random.seed(0)

MODEL_NAME = "openai/clip-vit-large-patch14"
MODEL_KEY  = MODEL_NAME.replace("/", "_")

IMAGE_EXTS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG",
              "*.tiff", "*.tif", "*.png", "*.PNG")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_video_frames(video_path: str) -> list:
    """Decode an mp4 into PIL Images, dropping the first and last frame."""
    frames = []
    container = av.open(video_path)
    for frame in container.decode(video=0):
        frames.append(frame.to_image())
    container.close()
    return frames[1:-1]


def embed_images(images: list, processor, model, device) -> np.ndarray:
    """Return (N, D) float32 embeddings for a list of PIL Images."""
    embeddings = []
    for img in images:
        inputs = processor(images=img.convert("RGB"), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=pixel_values)
            feat = model.visual_projection(vision_out.pooler_output)
        embeddings.append(feat.cpu().numpy()[0])
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main worker logic (always runs on cuda:0 within this process)
# ---------------------------------------------------------------------------

def main(args):
    raw_dir  = os.path.join(args.mosaic_root, "stimuli", "raw")
    save_dir = os.path.join(args.mosaic_root, "model_features", "clip_feats_viz")
    os.makedirs(save_dir, exist_ok=True)

    # ---- Gather all stimuli (deterministic sort so rank-slicing is consistent)
    video_paths = sorted(glob.glob(os.path.join(raw_dir, "*.mp4")))
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(glob.glob(os.path.join(raw_dir, ext)))
    image_paths.sort()

    n_gpus = len(args.gpus)

    # ---- Launcher: spawn one subprocess per GPU ---------------------------
    if n_gpus > 1 and args.rank == 0 and args.world_size == 1:
        print(f"Launching {n_gpus} workers on GPUs {args.gpus}")
        procs = []
        for rank, gpu_id in enumerate(args.gpus):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--mosaic_root", args.mosaic_root,
                   "--model_name",  args.model_name,
                   "--gpus",        str(gpu_id),
                   "--rank",        str(rank),
                   "--world_size",  str(n_gpus)]
            procs.append(subprocess.Popen(cmd, env=env))
        for p in procs:
            ret = p.wait()
            if ret != 0:
                print(f"  Warning: a worker exited with code {ret}")
        return

    # ---- Worker path (single GPU or called as subprocess) -----------------
    # CUDA_VISIBLE_DEVICES is already set in the env by the launcher (or the user).
    # Always use cuda:0 — it maps to the correct physical GPU via CUDA_VISIBLE_DEVICES.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_label = args.gpus[0]
    print(f"  Rank {args.rank}/{args.world_size} | GPU {gpu_label} | device {device}")

    model     = CLIPModel.from_pretrained(args.model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    model_key = args.model_name.replace("/", "_")

    # Slice file lists for this rank (round-robin for even distribution)
    my_videos = video_paths[args.rank::args.world_size]
    my_images = image_paths[args.rank::args.world_size]

    # ---- Videos -----------------------------------------------------------
    for video_path in tqdm(my_videos, desc=f"GPU{gpu_label} videos", position=0, leave=True):
        stem     = Path(video_path).stem
        out_path = os.path.join(save_dir, f"{stem}_model-{model_key}.npy")
        if os.path.isfile(out_path):
            continue
        frames = decode_video_frames(video_path)
        if len(frames) == 0:
            print(f"  Warning: no usable frames in {video_path}, skipping.")
            continue
        frame_embs = embed_images(frames, processor, model, device)
        np.save(out_path, frame_embs.mean(axis=0))

    # ---- Images -----------------------------------------------------------
    for image_path in tqdm(my_images, desc=f"GPU{gpu_label} images", position=1, leave=True):
        stem     = Path(image_path).stem
        out_path = os.path.join(save_dir, f"{stem}_model-{model_key}.npy")
        if os.path.isfile(out_path):
            continue
        image = Image.open(image_path).convert("RGB")
        emb   = embed_images([image], processor, model, device)
        np.save(out_path, emb[0])


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mosaic_root", type=str,
                        default="/data/vision/oliva/datasets/MOSAIC")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--gpus", type=int, nargs="+",
                        default=list(range(torch.cuda.device_count())) or [0],
                        help="GPU IDs to use (e.g. --gpus 0 1 2 3).")
    # Internal args used by subprocesses — do not set manually
    parser.add_argument("--rank",       type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
