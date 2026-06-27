"""
Generate UMAP embeddings from CLIP or DreamSim features across MOSAIC datasets.
Outputs umap_<model>_data.json for use with umap_viewer.html.

Embeddings are read from the unified model_features directory:
  <mosaic_root>/model_features/clip_feats_viz/        -- CLIP
  <mosaic_root>/model_features/dreamsim_feats_viz/   -- DreamSim

Each .npy file is named  {stimulus_stem}_model-{model_key}.npy
where stimulus_stem == Path(raw_stimulus_filename).stem.

Dataset and train/test/filtered labels are assigned by matching stimulus stems
against the mosaic JSON split files (train_naturalistic.json, etc.).
Stimuli shared across datasets (e.g. GOD/deeprecon ImageNet images) produce one
UMAP point per dataset so each dataset is fully represented when filtered in the
viewer.

Thumbnail image paths are resolved from the local compressed-stimuli directory
that was synced to S3 as stimuli_compressed/  (default:
  stimuli/stimuli_compressed_resized_quality-40_size-112/).

Usage:
    python generate_umap_data.py [--model clip|dreamsim]
                                 [--max_per_dataset 2000]
                                 [--mosaic_root /data/vision/oliva/datasets/MOSAIC]
                                 [--compressed_dir stimuli_compressed_resized_quality-40_size-112]
                                 [--output umap_clip_data.json]
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# Strips video-frame suffixes from compressed-thumbnail filenames so the dict
# key matches the bare mp4 stem used in split_map and embedding filenames.
# Handles three formats produced historically:
#   HAD new : _frame-0030_0061   (_frame-\d+_\d+)
#   HAD old : _frame30_61        (_frame\d+_\d+)   ← -? makes dash optional
#   BMD     : _45_90             (_\d+_\d+)
FRAME_SUFFIX_RE = re.compile(r"(_frame-?\d+_\d+|_\d+_\d+)$")

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
    Parse the three JSON split files and return:
        raw_stem → list of (dataset_tag, split_label)

    A stimulus shared across multiple datasets (e.g. GOD/deeprecon ImageNet
    images) gets one entry per dataset so the viewer can show it under each.
    Within a single dataset the first split encountered wins (no duplicates).

    raw_stem is Path(stim_id).stem, identical to the embedding filename stem.
    """
    # stem → {dataset_tag: split_label}  (ordered insertion, no duplicates per tag)
    split_map: dict[str, dict[str, str]] = defaultdict(dict)
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
            if tag not in split_map[stem]:  # first split per dataset wins
                split_map[stem][tag] = label
    # Convert to list of (tag, split) tuples, preserving insertion order
    return {stem: list(tag_split.items()) for stem, tag_split in split_map.items()}


def build_filtered_map(mosaic_root: str, split_map: dict) -> dict:
    """
    For stimuli that have embeddings but were excluded from all MOSAIC split
    files (i.e. not in split_map), use compiled_dataset_stiminfo.tsv to assign
    the correct dataset(s) with split_label = 'filtered'.

    Returns a dict with the same format as build_split_map:
        raw_stem → list of (dataset_tag, 'filtered')
    Only stems NOT already in split_map are included.
    """
    tsv_path = os.path.join(
        mosaic_root, "stimuli", "datasets_stiminfo", "compiled_dataset_stiminfo.tsv"
    )
    if not os.path.exists(tsv_path):
        print(f"  Warning: {tsv_path} not found, filtered stimuli will stay 'unknown'.")
        return {}

    tsv     = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    tt_cols = [c for c in tsv.columns if c.startswith("test_train_")]

    filtered = {}
    for _, row in tsv.iterrows():
        stem     = Path(str(row["filename"])).stem
        if stem in split_map:
            continue   # already assigned by JSON split files
        datasets = [c.replace("test_train_", "") for c in tt_cols if pd.notna(row[c])]
        if datasets:
            filtered[stem] = [(tag, "filtered") for tag in datasets]

    print(f"  Filtered (TSV fallback): {len(filtered):,} additional stems")
    return filtered


# ---------------------------------------------------------------------------
# Thumbnail lookup
# ---------------------------------------------------------------------------

def build_image_lookup(mosaic_root: str, compressed_dir: str) -> dict:
    """
    Build a lookup dict keyed by raw stimulus stem:
      stem → relative path under stimuli_compressed/  (as used in S3)

    compressed_dir is the local directory whose contents were synced to the
    S3 stimuli_compressed/ prefix (e.g. stimuli_compressed_resized_quality-40_size-112).
    Files there already use bare mp4 stems as filenames; FRAME_SUFFIX_RE is kept
    as a safety net for any legacy formats still present.
    """
    stimuli_root = Path(mosaic_root) / "stimuli"
    src = stimuli_root / compressed_dir
    if not src.exists():
        raise FileNotFoundError(f"Compressed stimuli directory not found: {src}")

    compressed = {}
    for f in sorted(src.iterdir()):
        compressed[f.stem] = f"stimuli_compressed/{f.name}"

    print(f"  stimuli_compressed : {len(compressed):,} keys  (from {compressed_dir})")
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
    print(f"  {len(split_map):,} unique stimuli mapped from JSON split files")

    print("Loading TSV fallback for filtered stimuli...")
    filtered_map = build_filtered_map(mosaic_root, split_map)
    split_map.update(filtered_map)

    # Count total (dataset, split) pairs across all stems
    all_pairs = [pair for pairs in split_map.values() for pair in pairs]
    print(f"  {len(split_map):,} total stems")
    print(" ", Counter(split for _, split in all_pairs))

    # ---- Build thumbnail lookup -------------------------------------------
    print("Building thumbnail lookup...")
    compressed = build_image_lookup(mosaic_root, args.compressed_dir)

    # ---- Gather .npy files, optionally capped per dataset -----------------
    print(f"\nScanning {emb_dir} ...")
    all_files = list(emb_dir.glob("*.npy"))
    print(f"  Found {len(all_files):,} embedding files")

    if args.max_per_dataset:
        # Group by primary (first) dataset tag, then sample within each group
        files_by_tag = defaultdict(list)
        for fpath in all_files:
            stem    = fpath.name.split("_model-")[0]
            entries = split_map.get(stem) or [("unknown", "filtered")]
            tag     = entries[0][0]   # primary dataset
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
        stem    = fpath.name.split("_model-")[0]
        entries = split_map.get(stem) or [("unknown", "filtered")]
        img     = get_thumbnail(stem, compressed)

        embeddings.append(np.load(fpath).flatten())
        # Store all (dataset, split) pairs; one point per pair in the output
        labels.append({"entries": entries, "img": img, "name": stem})
        for (tag, split) in entries:
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
    # Expand shared stimuli: one point per (dataset, split) pair so that every
    # dataset shows the stimulus when selected in the viewer.
    points = []
    for i, lbl in enumerate(labels):
        for (tag, split) in lbl["entries"]:
            points.append({
                "x":       round(float(coords[i, 0]), 4),
                "y":       round(float(coords[i, 1]), 4),
                "dataset": tag,
                "split":   split,
                "name":    lbl["name"],
                "img":     lbl["img"],
            })

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
        "--compressed_dir", type=str,
        default="stimuli_compressed_resized_quality-40_size-112",
        help="Subdirectory of <mosaic_root>/stimuli/ whose contents were synced "
             "to S3 as stimuli_compressed/. Default matches the quality-40 upload.",
    )
    parser.add_argument(
        "--max_per_dataset", type=int, default=None,
        help="Max stimuli per dataset. Default: use all.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path. Default: umap_<model>_data.json next to this script.",
    )
    args = parser.parse_args()
    if args.output is None:
        stem        = f"umap_{args.model}_data.json"
        args.output = os.path.join(os.path.dirname(os.path.abspath(__file__)), stem)
    main(args)
