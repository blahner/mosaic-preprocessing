"""
Generate UMAP embeddings from CLIP or DreamSim features across MOSAIC datasets.
Outputs umap_data.json (or umap_dreamsim_data.json) for use with umap_viewer.html.

Embeddings are read from the unified model_features directory:
  <mosaic_root>/model_features/clip_feats_viz/        -- CLIP
  <mosaic_root>/model_features/dreamsim_feats_viz/   -- DreamSim

Each .npy file is named  {stimulus_stem}_model-{model_key}.npy
where stimulus_stem == Path(raw_stimulus_filename).stem.

Dataset and train/test/filtered labels are assigned by matching stimulus stems
against the mosaic JSON split files (train_naturalistic.json, etc.).

Thumbnail image paths are resolved from:
  stimuli/stimuli_compressed_quality-95_size-224/  -- all stimuli (as served from S3)

Usage:
    python generate_umap_data.py [--model clip|dreamsim]
                                 [--max_per_dataset 2000]
                                 [--mosaic_root /data/vision/oliva/datasets/MOSAIC]
                                 [--output umap_data.json]
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import umap

np.random.seed(42)

# Dataset tags as they appear in the JSON split file keys
DATASET_TAGS = ["NSD", "BOLD5000", "GOD", "NOD", "deeprecon", "THINGS", "BMD", "HAD"]

KEY_RE = re.compile(r"sub-\d+_(.+?)_stimulus-(.+)")


# ---------------------------------------------------------------------------
# Build split map  (raw_stem → (tag, split_label))
# ---------------------------------------------------------------------------

def build_split_map(mosaic_root: str) -> dict:
    """
    Parse the three JSON split files and return a flat dict:
        raw_stem → (dataset_tag, split_label)

    raw_stem is Path(stim_id).stem, i.e. the raw filename without extension,
    which is identical to the embedding filename stem used by both
    clip_embeddings_viz.py and dreamsim_embeddings_viz.py.
    """
    split_map = {}
    for fname, label in [
        ("train_naturalistic.json", "train_nat"),
        ("test_naturalistic.json",  "test_nat"),
        ("test_artificial.json",    "test_art"),
    ]:
        fpath = os.path.join(mosaic_root, fname)
        if not os.path.exists(fpath):
            print(f"  Warning: {fname} not found, skipping.")
            continue
        entries = json.load(open(fpath))
        for entry in entries:
            raw_key = list(entry.keys())[0]
            m = KEY_RE.match(raw_key)
            if not m:
                continue
            tag, stim_id = m.group(1), m.group(2)
            stem = Path(stim_id).stem       # strip extension → raw file stem
            if stem not in split_map:       # first occurrence wins
                split_map[stem] = (tag, label)
    return split_map


# ---------------------------------------------------------------------------
# Thumbnail lookup
# ---------------------------------------------------------------------------

def build_image_lookup(mosaic_root: str) -> dict:
    """
    Build a lookup dict keyed by raw stimulus stem:
      compressed : stem → relative path under stimuli/stimuli_compressed/

    All thumbnails (images and videos) are served from stimuli_compressed/ in S3.
    The relative paths are what the web viewer uses to request thumbnails.
    """
    stimuli_root = Path(mosaic_root) / "stimuli"

    compressed = {}
    for f in (stimuli_root / "stimuli_compressed_quality-95_size-224").iterdir():
        compressed[f.stem] = f"stimuli_compressed/{f.name}"

    print(f"  stimuli_compressed : {len(compressed):,} files")
    return compressed


def get_thumbnail(stem: str, compressed: dict):
    """Return a relative thumbnail path for the web viewer, or None."""
    return compressed.get(stem)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    mosaic_root = args.mosaic_root

    if args.model == "dreamsim":
        model_label  = "dreamsim"
        emb_dir_name = "dreamsim_feats_viz"
    else:
        model_label  = "openai_clip-vit-large-patch14"
        emb_dir_name = "clip_feats_viz"

    emb_dir = Path(mosaic_root) / "model_features" / emb_dir_name
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")

    # ---- Build split map --------------------------------------------------
    print("Parsing JSON split files...")
    split_map = build_split_map(mosaic_root)
    print(f"  {len(split_map):,} unique stimuli mapped")
    print(" ", Counter(v[1] for v in split_map.values()))

    # ---- Build thumbnail lookup -------------------------------------------
    print("Building thumbnail lookup...")
    compressed = build_image_lookup(mosaic_root)

    # ---- Gather .npy files, optionally capped per dataset -----------------
    print(f"\nScanning {emb_dir} ...")
    all_files = list(emb_dir.glob("*.npy"))
    print(f"  Found {len(all_files):,} embedding files")

    if args.max_per_dataset:
        # Group by dataset tag, then sample within each group
        files_by_tag = defaultdict(list)
        for fpath in all_files:
            stem = fpath.name.split("_model-")[0]
            tag  = split_map.get(stem, ("unknown",))[0]
            files_by_tag[tag].append(fpath)

        sampled = []
        for tag, files in sorted(files_by_tag.items()):
            if len(files) > args.max_per_dataset:
                chosen = np.random.choice(len(files), args.max_per_dataset, replace=False)
                sampled.extend(files[i] for i in chosen)
            else:
                sampled.extend(files)
        all_files = sampled
        print(f"  After capping at {args.max_per_dataset} per dataset: "
              f"{len(all_files):,} files")

    # ---- Load embeddings --------------------------------------------------
    print("\nLoading embeddings...")
    embeddings, labels = [], []
    counts = Counter()

    for fpath in all_files:
        stem  = fpath.name.split("_model-")[0]
        entry = split_map.get(stem)
        tag   = entry[0] if entry else "unknown"
        split = entry[1] if entry else "filtered"
        img   = get_thumbnail(stem, compressed)

        embeddings.append(np.load(fpath).flatten())
        labels.append({"dataset": tag, "split": split, "img": img, "name": stem})
        counts[(tag, split)] += 1

    for tag in DATASET_TAGS + ["unknown"]:
        tag_counts = {s: counts[(tag, s)] for s in ["train_nat", "test_nat", "test_art", "filtered"]
                      if counts[(tag, s)] > 0}
        if tag_counts:
            print(f"  {tag}: {tag_counts}")

    # ---- UMAP -------------------------------------------------------------
    X = np.stack(embeddings)
    print(f"\nEmbedding matrix: {X.shape}")

    print("\nRunning UMAP...")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        random_state=42, verbose=True,
    )
    coords = reducer.fit_transform(X)
    print("Done.")

    # ---- Build output JSON ------------------------------------------------
    points = [
        {
            "x":       round(float(coords[i, 0]), 4),
            "y":       round(float(coords[i, 1]), 4),
            "dataset": labels[i]["dataset"],
            "split":   labels[i]["split"],
            "name":    labels[i]["name"],
            "img":     labels[i]["img"],
        }
        for i in range(len(labels))
    ]

    out = {
        "meta": {
            "model":           model_label,
            "n_points":        len(points),
            "max_per_dataset": args.max_per_dataset,
            "datasets":        sorted(set(p["dataset"] for p in points)),
            "splits":          ["train_nat", "test_nat", "test_art", "filtered"],
        },
        "points": points,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"\nSaved {len(points):,} points to {args.output}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mosaic_root", type=str,
        default=os.path.join(
            os.getenv("DATASETS_ROOT", "/data/vision/oliva/datasets"), "MOSAIC"
        ),
        help="Root path to the MOSAIC dataset.",
    )
    parser.add_argument(
        "--model", type=str, default="clip", choices=["clip", "dreamsim"],
        help="Which embeddings to use (default: clip).",
    )
    parser.add_argument(
        "--max_per_dataset", type=int, default=None,
        help="Max stimuli per dataset. Default: use all.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path. Default: umap_data.json or umap_dreamsim_data.json "
             "next to this script.",
    )
    args = parser.parse_args()
    if args.output is None:
        stem        = "umap_dreamsim_data.json" if args.model == "dreamsim" else "umap_data.json"
        args.output = os.path.join(os.path.dirname(os.path.abspath(__file__)), stem)
    main(args)
