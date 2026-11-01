#!/usr/bin/env bash
#
# download_stimuli.sh — Download all MOSAIC stimulus sets
#
# Usage:
#   source .env && bash src/stimulusSetPreparation/download_stimuli.sh
#
# Prerequisites:
#   - DATASETS_ROOT and PROJECT_ROOT must be set (define them in your .env file)
#   - AWS CLI installed and configured for unauthenticated S3 access
#     (used for NOD and NSD); `uv tool install awscli` works
#   - uv installed, and `uv sync` already run in PROJECT_ROOT (the NSD step
#     runs a small pandas snippet through `uv run`)
#
# What this script does:
#   - Automatically downloads stimuli that are hosted on public S3 (NOD, NSD) and the pinned HAD v1.1.1 videos
#     plus BOLD5000 (Dropbox) and the Deeprecon artificial/letter images +
#     stimulus maps (figshare)
#   - Prints step-by-step manual instructions for datasets that require
#     account registration, request forms, browser downloads, or custom tooling
#     (BMD, GOD/Deeprecon natural images, THINGS)
#   - Skips any dataset whose stimuli are already present
#
# After this script completes, follow the README step 3b to extract video
# frames and preprocess NSD synthetic stimuli before running step 3c.

MANUAL_DOWNLOADS=()
SKIPPED_DOWNLOADS=()
COMPLETED_DOWNLOADS=()
FAILED_DOWNLOADS=()

# ── terminal colors ───────────────────────────────────────────────────────────

if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; NC=''
fi

# ── helpers ───────────────────────────────────────────────────────────────────

section() {
    echo
    echo -e "${BOLD}${BLUE}=== $1 ===${NC}"
}

info()    { echo -e "  $1"; }
ok()      { echo -e "  ${GREEN}✓${NC} $1"; }
warn()    { echo -e "  ${YELLOW}!${NC} $1"; }
err()     { echo -e "  ${RED}✗${NC} $1"; }

# Returns 0 (true) if $1 exists as a directory and contains at least $2 files.
has_files() {
    local dir="$1" min="${2:-1}"
    [ -d "$dir" ] && \
        [ "$(find "$dir" -type f 2>/dev/null | head -n "$min" | wc -l | tr -d ' ')" -ge "$min" ]
}

check_env() {
    if [ -z "${DATASETS_ROOT}" ]; then
        echo -e "${RED}ERROR:${NC} DATASETS_ROOT is not set."
        echo "  Source your .env file before running this script:"
        echo "    source .env && bash src/stimulusSetPreparation/download_stimuli.sh"
        exit 1
    fi
    # PROJECT_ROOT locates the uv project used for the NSD post-processing step
    if [ -z "${PROJECT_ROOT}" ]; then
        echo -e "${RED}ERROR:${NC} PROJECT_ROOT is not set."
        echo "  Source your .env file before running this script:"
        echo "    source .env && bash src/stimulusSetPreparation/download_stimuli.sh"
        exit 1
    fi
    if ! command -v aws &>/dev/null; then
        warn "AWS CLI not found. Datasets that require S3 (NOD, NSD) will be skipped."
        NO_AWS=1
    fi
}

# ── BOLD5000 ──────────────────────────────────────────────────────────────────
#
# Downloaded from the Dropbox link on https://bold5000-dataset.github.io/website/download.html
# (BOLD5000_Stimuli.zip, ~1.2 GB). The zip's top-level BOLD5000_Stimuli/ folder is
# stripped so its contents land directly in stimuli_metadata/.
# Expected layout after download:
#   $DATASETS_ROOT/BOLD5000/derivatives/stimuli_metadata/
#     Scene_Stimuli/repeated_stimuli_113_list.txt   (test-set list, used by extract_b5000_stiminfo.py)
#     Scene_Stimuli/Presented_Stimuli/COCO/*.jpg       (2000 images)
#     Scene_Stimuli/Presented_Stimuli/ImageNet/*.JPEG  (1916 images)
#     Scene_Stimuli/Presented_Stimuli/Scene/*.jpg      (1000 images)
#     Scene_Stimuli/Original_Images/, Image_Labels/, Stimuli_Presentation_Lists/, Localizer_Stimuli/

BOLD5000_STIMULI_URL="https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1"

download_bold5000() {
    section "BOLD5000"
    local root="$DATASETS_ROOT/BOLD5000/derivatives/stimuli_metadata"
    local dest="$root/Scene_Stimuli/Presented_Stimuli"
    if has_files "$dest" 4916 && [ -f "$root/Scene_Stimuli/repeated_stimuli_113_list.txt" ]; then
        ok "BOLD5000 stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("BOLD5000")
        return
    fi
    info "Downloading BOLD5000 stimuli from Dropbox (~1.2 GB)..."
    mkdir -p "$root"
    local zip="$root/BOLD5000_Stimuli.zip"
    if curl -fL --progress-bar -o "$zip" "$BOLD5000_STIMULI_URL" \
        && unzip -q -o "$zip" -d "$root" \
        && cp -a "$root/BOLD5000_Stimuli/." "$root/" \
        && rm -rf "$root/BOLD5000_Stimuli" "$zip" \
        && has_files "$dest" 4916; then
        ok "BOLD5000 download complete."
        COMPLETED_DOWNLOADS+=("BOLD5000")
    else
        err "BOLD5000 download failed. Download BOLD5000_Stimuli.zip manually from:"
        info "  https://bold5000-dataset.github.io/website/download.html"
        info "and extract the contents of its BOLD5000_Stimuli/ folder into:"
        info "  $root/"
        FAILED_DOWNLOADS+=("BOLD5000")
    fi
}

# ── BOLD Moments Dataset (BMD) ────────────────────────────────────────────────
#
# stimulus_set.zip (~3 GB) is public to download from https://boldmomentsdataset.csail.mit.edu/stimuli_metadata/
# but password-protected: the password is at the bottom of stimuli_access.txt in the same folder,
# after the Moments in Time terms of use, so the user must read the terms and unzip it themselves.
# annotations.json (BMD video number -> Moments in Time filename) is public on OpenNeuro.
# Expected layout after download + unzip:
#   $DATASETS_ROOT/BOLDMomentsDataset/derivatives/stimuli_metadata/
#     annotations.json
#     mp4_h264/*.mp4               (1102 mp4 videos, named 0001.mp4 .. 1102.mp4)
#     frames/XXXX/XXXX_<frame>_<nframes>.jpg   (all frames of each video)
#     frames_middle/XXXX_<frame>_<nframes>.jpg (1102 middle frames)
#     stimuli/{train,test}/*.mp4   (same videos, split by BMD's train/test)

BMD_STIMULI_URL="https://boldmomentsdataset.csail.mit.edu/stimuli_metadata"
BMD_ANNOTATIONS_URL="https://s3.amazonaws.com/openneuro.org/ds005165/derivatives/stimuli_metadata/annotations.json"

download_bmd() {
    section "BOLD Moments Dataset (BMD)"
    local dest="$DATASETS_ROOT/BOLDMomentsDataset/derivatives/stimuli_metadata"
    mkdir -p "$dest"

    if [ ! -f "$dest/annotations.json" ]; then
        info "Downloading BMD annotations.json from OpenNeuro..."
        curl -fsSL -o "$dest/annotations.json" "$BMD_ANNOTATIONS_URL" \
            || { err "Failed to download annotations.json"; rm -f "$dest/annotations.json"; }
    fi

    if has_files "$dest/mp4_h264" 1102 && has_files "$dest/frames_middle" 1102 \
        && [ -f "$dest/annotations.json" ]; then
        ok "BMD stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("BMD")
        return
    fi

    local zip="$dest/stimulus_set.zip"
    if [ ! -f "$zip" ]; then
        info "Downloading BMD stimulus_set.zip (~3 GB)..."
        curl -fL --progress-bar -o "$zip.part" "$BMD_STIMULI_URL/stimulus_set.zip" && mv "$zip.part" "$zip" \
            || { err "Failed to download stimulus_set.zip"; rm -f "$zip.part"; FAILED_DOWNLOADS+=("BMD"); return; }
    fi
    warn "BMD stimulus_set.zip is password-protected and must be unzipped manually."
    info "1. Read the terms of use; the password is at the end:"
    info "     $BMD_STIMULI_URL/stimuli_access.txt"
    info "2. Unzip (you'll be prompted for the password) and move its contents into place:"
    info "     unzip $zip -d $dest"
    info "     mv $dest/stimulus_set/* $dest/ && rmdir $dest/stimulus_set && rm $zip"
    MANUAL_DOWNLOADS+=("BMD")
}

# ── Generic Object Decoding + Deeprecon ───────────────────────────────────────
#
# GOD and Deeprecon share the same natural task stimuli (1250 ImageNet images).
# These are stored ONCE, under GOD; deeprecon's copy step reads them from there.
# Deeprecon additionally has artificial shape/letter images (public on figshare)
# and stimulus_id -> filename maps for all four of its stimulus sets.
# Expected layout:
#   $DATASETS_ROOT/GenericObjectDecoding/derivatives/stimuli_metadata/images/
#     test/*.JPEG             (50 images)
#     training/*.JPEG         (1200 images)
#     image_test_id.csv       (stimulus_id -> filename map)
#     image_training_id.csv
#   $DATASETS_ROOT/deeprecon/derivatives/stimuli_metadata/images/
#     ArtificialImage/*.tiff  (40 images)
#     LetterImage/*.tif       (10 images)
#     NaturalImageTraining.tsv, NaturalImageTest.tsv,
#     ArtificialImage.tsv, LetterImage.tsv   (stimulus_id -> filename maps)

# Deep Image Reconstruction figshare (https://figshare.com/articles/dataset/Deep_Image_Reconstruction/7033577)
# file IDs, pinned so the download doesn't depend on figshare's listing order.
DEEPRECON_FIGSHARE_URL="https://ndownloader.figshare.com/files"
declare -A DEEPRECON_FIGSHARE_TSV=(
    [NaturalImageTraining.tsv]=14876741   # stimulus_NaturalImageTraining.tsv
    [NaturalImageTest.tsv]=14876738       # stimulus_NaturalImageTest.tsv
    [ArtificialImage.tsv]=14876798        # stimulus_ArtificialImage.tsv
    [LetterImage.tsv]=14876732            # stimulus_LetterImage.tsv
)
declare -A DEEPRECON_FIGSHARE_ZIP=(
    [ArtificialImage]=14876801            # stimulus_ArtificialImage.zip -> ArtificialImage/*.tiff
    [LetterImage]=14876735                # stimulus_LetterImage.zip     -> LetterImage/*.tif
)

download_god_deeprecon() {
    section "Generic Object Decoding / Deeprecon"
    local dest_god="$DATASETS_ROOT/GenericObjectDecoding/derivatives/stimuli_metadata/images"
    local dest_dr="$DATASETS_ROOT/deeprecon/derivatives/stimuli_metadata/images"

    # -- GOD/deeprecon natural images: copyrighted, only available by request --
    if has_files "$dest_god/test" 50 && has_files "$dest_god/training" 1200 \
        && [ -f "$dest_god/image_test_id.csv" ] && [ -f "$dest_god/image_training_id.csv" ]; then
        ok "GOD/Deeprecon natural images already present — skipping."
        SKIPPED_DOWNLOADS+=("GOD-natural-images")
    else
        warn "GOD/Deeprecon natural images (1250 ImageNet images) require a request form."
        info "1. Fill out the Kamitani Lab request form (linked from"
        info "   https://github.com/KamitaniLab/GenericObjectDecoding):"
        info "     https://forms.gle/ujvA34948Xg49jdn9"
        info "   A reply arrives by email right away with a Google Drive link and a zip password."
        info "2. Download the zip using the file ID from the Drive link"
        info "   (https://drive.google.com/file/d/<FILE_ID>/view):"
        info "     uv tool run gdown <FILE_ID> -O \$TMP/"
        info "3. Unzip it (you'll be prompted for the password). It extracts to images/ containing"
        info "   test/, training/, image_test_id.csv and image_training_id.csv. Move it into place:"
        info "     unzip \$TMP/images.zip -d \$TMP/"
        info "     mkdir -p $(dirname "$dest_god")"
        info "     mv \$TMP/images $dest_god"
        MANUAL_DOWNLOADS+=("GOD-natural-images")
    fi

    # -- Deeprecon artificial/letter images + stimulus maps: public on figshare --
    local need=0 name
    has_files "$dest_dr/ArtificialImage" 40 || need=1
    has_files "$dest_dr/LetterImage" 10 || need=1
    for name in "${!DEEPRECON_FIGSHARE_TSV[@]}"; do
        [ -f "$dest_dr/$name" ] || need=1
    done
    if [ "$need" -eq 0 ]; then
        ok "Deeprecon artificial/letter images and stimulus maps already present — skipping."
        SKIPPED_DOWNLOADS+=("Deeprecon-figshare")
        return
    fi

    info "Downloading Deeprecon artificial/letter images and stimulus maps from figshare..."
    mkdir -p "$dest_dr"
    local failed=0
    for name in "${!DEEPRECON_FIGSHARE_TSV[@]}"; do
        [ -f "$dest_dr/$name" ] && continue
        curl -fsSL -o "$dest_dr/$name" "$DEEPRECON_FIGSHARE_URL/${DEEPRECON_FIGSHARE_TSV[$name]}" \
            || { err "Failed to download $name"; rm -f "$dest_dr/$name"; failed=1; }
    done
    for name in "${!DEEPRECON_FIGSHARE_ZIP[@]}"; do
        has_files "$dest_dr/$name" 1 && continue
        local zip="$dest_dr/$name.zip"
        curl -fsSL -o "$zip" "$DEEPRECON_FIGSHARE_URL/${DEEPRECON_FIGSHARE_ZIP[$name]}" \
            && unzip -q -o "$zip" -d "$dest_dr" && rm "$zip" \
            || { err "Failed to download/extract $name images"; failed=1; }
    done

    if [ "$failed" -eq 0 ] && has_files "$dest_dr/ArtificialImage" 40 && has_files "$dest_dr/LetterImage" 10; then
        ok "Deeprecon figshare download complete."
        COMPLETED_DOWNLOADS+=("Deeprecon-figshare")
    else
        err "Deeprecon figshare download incomplete. Files are listed at:"
        info "  https://figshare.com/articles/dataset/Deep_Image_Reconstruction/7033577"
        FAILED_DOWNLOADS+=("Deeprecon-figshare")
    fi
}

# ── Human Actions Dataset (HAD) ───────────────────────────────────────────────
#
# Expected layout after download + step 3b preprocessing:
#   $DATASETS_ROOT/HumanActionsDataset/Nifti/stimuli/
#     <category>/*.mp4     (180 categories × 120 videos each)
#   derivatives/stimuli_metadata/frames_middle/*.jpg  (created by step 3b)
#   derivatives/stimuli_metadata/frames/*/            (created by step 3b)

# The stimuli are pinned to the OpenNeuro ds004488 v1.1.1 snapshot used for MOSAIC
# (category folders and filenames contain spaces, e.g. "Arm wrestling/v_Arm wrestling_id_...mp4").
# Each file is fetched from the versioned OpenNeuro S3 bucket by the versionId recorded in
# had_stimuli_v1.1.1_manifest.tsv and checked against its md5, so no AWS CLI is needed.

HAD_MANIFEST="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/HumanActionsDataset/download/had_stimuli_v1.1.1_manifest.tsv"

download_had() {
    section "Human Actions Dataset (HAD)"
    local dest="$DATASETS_ROOT/HumanActionsDataset/Nifti/stimuli"
    if has_files "$dest" 21600; then
        ok "HAD stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("HAD")
        return
    fi
    info "Downloading the 21600 HAD stimulus videos (OpenNeuro ds004488 v1.1.1, ~2.2 GB)..."
    mkdir -p "$dest"
    HAD_DEST="$dest" HAD_MANIFEST="$HAD_MANIFEST" uv run --project "${PROJECT_ROOT}" python - <<'PYEOF' 2>&1 | while IFS= read -r line; do info "  $line"; done
import csv, hashlib, os, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
dest, manifest = os.environ["HAD_DEST"], os.environ["HAD_MANIFEST"]
rows = list(csv.DictReader(open(manifest), delimiter="\t"))

def fetch(r):
    path = os.path.join(dest, r["path"])
    if os.path.exists(path) and os.path.getsize(path) == int(r["size"]):
        return None  # already downloaded (safe to re-run after an interruption)
    url = ("https://s3.amazonaws.com/openneuro.org/ds004488/stimuli/"
           + urllib.parse.quote(r["path"]) + "?versionId=" + urllib.parse.quote(r["version_id"]))
    for _ in range(3):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            if hashlib.md5(data).hexdigest() == r["md5"]:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(data)
                return None
        except Exception:
            pass
    return r["path"]

with ThreadPoolExecutor(16) as ex:
    failed = [p for p in ex.map(fetch, rows) if p]
print(f"{len(rows) - len(failed)}/{len(rows)} videos present and verified")
for p in failed[:10]:
    print(f"failed: {p}")
sys.exit(1 if failed else 0)
PYEOF
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        ok "HAD download complete."
        COMPLETED_DOWNLOADS+=("HAD")
    else
        err "HAD download incomplete. Re-run this script to retry the remaining videos."
        FAILED_DOWNLOADS+=("HAD")
    fi
}

# ── THINGS ────────────────────────────────────────────────────────────────────
#
# images_THINGS.zip (~5 GB, all 26,107 THINGS images) is public to download from the
# THINGS OSF project (https://osf.io/jum2f/) but password-protected: the password is in
# password_images.txt on the same project, after the license terms, so the user must
# read the terms and unzip it themselves. Only the 8740 images shown in THINGS-fMRI
# (listed in THINGS_fmri/download/THINGS_fmri_filenames.txt) are then copied out.
# Expected layout after download + unzip + re-running this script:
#   $DATASETS_ROOT/THINGS_fmri/derivatives/stimuli_metadata/
#     images/<concept>/*.jpg        (full THINGS database, extracted from the zip)
#     experimental_images/*.jpg     (the 8740 THINGS-fMRI images)

THINGS_IMAGES_URL="https://osf.io/download/rdxy2/"         # images_THINGS.zip
THINGS_PASSWORD_URL="https://osf.io/download/j6a3m/"       # password_images.txt
THINGS_FILENAMES="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/THINGS_fmri/download/THINGS_fmri_filenames.txt"

download_things() {
    section "THINGS"
    local root="$DATASETS_ROOT/THINGS_fmri/derivatives/stimuli_metadata"
    local dest="$root/experimental_images"
    if has_files "$dest" 8740; then
        ok "THINGS stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("THINGS")
        return
    fi

    # full database extracted -> copy out the experimental images
    if has_files "$root/images" 8740; then
        info "Copying the 8740 THINGS-fMRI images into experimental_images/..."
        mkdir -p "$dest"
        local f concept missing=0
        while IFS= read -r f; do
            [ -z "$f" ] && continue
            concept="${f%_*}"   # e.g. coat_rack_13s.jpg -> coat_rack
            cp -n "$root/images/$concept/$f" "$dest/" 2>/dev/null || missing=$((missing+1))
        done < "$THINGS_FILENAMES"
        if [ "$missing" -eq 0 ] && has_files "$dest" 8740; then
            ok "THINGS experimental images ready."
            COMPLETED_DOWNLOADS+=("THINGS")
        else
            err "$missing THINGS-fMRI images were not found under $root/images/."
            FAILED_DOWNLOADS+=("THINGS")
        fi
        return
    fi

    local zip="$root/images_THINGS.zip"
    if [ ! -f "$zip" ]; then
        info "Downloading images_THINGS.zip from OSF (~5 GB)..."
        mkdir -p "$root"
        curl -fL --progress-bar -o "$zip.part" "$THINGS_IMAGES_URL" && mv "$zip.part" "$zip" \
            || { err "Failed to download images_THINGS.zip"; rm -f "$zip.part"; FAILED_DOWNLOADS+=("THINGS"); return; }
    fi
    warn "images_THINGS.zip is password-protected and must be unzipped manually."
    info "1. Read the license terms; the password is at the end:"
    info "     $THINGS_PASSWORD_URL  (password_images.txt on https://osf.io/jum2f/)"
    info "2. Unzip (you'll be prompted for the password) and rename the extracted folder:"
    info "     unzip $zip -d $root"
    info "     mv $root/object_images $root/images && rm $zip"
    info "3. Re-run this script to copy the 8740 THINGS-fMRI images into experimental_images/."
    MANUAL_DOWNLOADS+=("THINGS")
}

# ── Natural Object Dataset (NOD) ──────────────────────────────────────────────
#
# Expected layout:
#   $DATASETS_ROOT/NaturalObjectDataset/Nifti/stimuli/
#     imagenet/<synset>/*.JPEG  (ImageNet images by synset folder)
#     coco/*.jpg                (COCO images)
#   Total: ~57120 images

download_nod() {
    section "Natural Object Dataset (NOD)"
    local dest="$DATASETS_ROOT/NaturalObjectDataset/Nifti/stimuli"
    if has_files "$dest" 100; then
        ok "NOD stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("NOD")
        return
    fi
    if [ -n "${NO_AWS}" ]; then
        warn "AWS CLI not available — skipping automated NOD download."
        info "Install the AWS CLI, then re-run this script, or download manually:"
        info "  aws s3 sync --no-sign-request s3://openneuro.org/ds004496/stimuli/ \\"
        info "    $dest/"
        MANUAL_DOWNLOADS+=("NOD")
        return
    fi
    info "Downloading NOD stimuli from OpenNeuro (~57k images)..."
    mkdir -p "$dest"
    if aws s3 sync --no-sign-request s3://openneuro.org/ds004496/stimuli/ "$dest/"; then
        ok "NOD download complete."
        COMPLETED_DOWNLOADS+=("NOD")
    else
        err "NOD download failed. Check your network connection and retry."
        FAILED_DOWNLOADS+=("NOD")
    fi
}

# ── Natural Scenes Dataset (NSD) ──────────────────────────────────────────────
#
# Expected layout:
#   $DATASETS_ROOT/NaturalScenesDataset/derivatives/stimuli_metadata/
#     nsd_stimuli.hdf5               (~39 GB — all 73000 COCO images)
#     nsdsynthetic_stimuli.hdf5      (~640 MB — needed by step 3b notebook)
#     nsd_stim_info_merged.csv
#     notshown.tsv                   (derived from nsd_stim_info_merged.csv)
#     annotations_trainval2017/
#       annotations/
#         instances_train2017.json
#         instances_val2017.json
#     nsdsynthetic_jpg/              (created by step 3b: save_nsdsynthetic_stimuli.ipynb)

download_nsd() {
    section "Natural Scenes Dataset (NSD)"
    local dest="$DATASETS_ROOT/NaturalScenesDataset/derivatives/stimuli_metadata"
    if has_files "$dest" 3 && [ -f "$dest/nsd_stimuli.hdf5" ]; then
        ok "NSD stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("NSD")
        return
    fi
    if [ -n "${NO_AWS}" ]; then
        warn "AWS CLI not available — skipping automated NSD download."
        info "Install the AWS CLI, then re-run this script, or see manual commands below."
        MANUAL_DOWNLOADS+=("NSD")
        _nsd_manual_instructions "$dest"
        return
    fi
    info "Downloading NSD stimuli from S3 (~40 GB total — this will take a while)..."
    mkdir -p "$dest"

    local failed=0

    # S3 key confirmed: nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5
    info "  [1/5] nsd_stimuli.hdf5 (~39 GB)"
    aws s3 cp --no-sign-request \
        s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
        "$dest/nsd_stimuli.hdf5" || { err "Failed to download nsd_stimuli.hdf5"; failed=1; }

    # S3 key confirmed: nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_stimuli.hdf5
    info "  [2/5] nsdsynthetic_stimuli.hdf5 (~640 MB, needed for step 3b)"
    aws s3 cp --no-sign-request \
        s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_stimuli.hdf5 \
        "$dest/nsdsynthetic_stimuli.hdf5" || { err "Failed to download nsdsynthetic_stimuli.hdf5"; failed=1; }

    # S3 key confirmed: nsddata/experiments/nsd/nsd_stim_info_merged.csv
    info "  [3/5] nsd_stim_info_merged.csv"
    aws s3 cp --no-sign-request \
        s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv \
        "$dest/nsd_stim_info_merged.csv" || { err "Failed to download nsd_stim_info_merged.csv"; failed=1; }

    # notshown.tsv is not distributed by NSD — derive it from nsd_stim_info_merged.csv.
    # It lists the 1-indexed nsdIds of images never shown to any subject (all trial counts = 0).
    info "  [4/5] notshown.tsv (derived from nsd_stim_info_merged.csv)"
    if [ -f "$dest/nsd_stim_info_merged.csv" ]; then
        NSD_DEST="$dest" uv run --project "${PROJECT_ROOT}" python - <<'PYEOF' 2>&1 | while IFS= read -r line; do info "    $line"; done
import os, pandas as pd
dest = os.environ.get("NSD_DEST")
csv_path = os.path.join(dest, "nsd_stim_info_merged.csv")
df = pd.read_csv(csv_path)
# subject columns track how many times each image was shown per subject
subj_cols = [c for c in df.columns if c.startswith("subject") and c[7:].isdigit()]
not_shown = df.index[df[subj_cols].sum(axis=1) == 0].tolist()
# nsdId in the CSV is 0-indexed; notshown.tsv uses 1-indexed nsdIds
not_shown_1indexed = [i + 1 for i in not_shown]
out_path = os.path.join(dest, "notshown.tsv")
pd.DataFrame(not_shown_1indexed).to_csv(out_path, index=False, header=False)
print(f"Wrote {len(not_shown_1indexed)} not-shown image IDs to {out_path}")
PYEOF
    else
        warn "nsd_stim_info_merged.csv not available — cannot derive notshown.tsv yet."
        warn "Re-run this script after step [3/5] succeeds to generate notshown.tsv."
        failed=1
    fi

    # COCO annotations come from the COCO website, not the NSD S3 bucket.
    info "  [5/5] COCO 2017 annotations"
    local coco_zip="$dest/annotations_trainval2017.zip"
    local coco_url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    if [ ! -f "$dest/annotations_trainval2017/annotations/instances_train2017.json" ]; then
        curl -L --progress-bar -o "$coco_zip" "$coco_url" && \
            unzip -q "$coco_zip" -d "$dest/annotations_trainval2017" && \
            rm "$coco_zip" || { err "Failed to download COCO annotations"; failed=1; }
    else
        ok "COCO annotations already present — skipping."
    fi

    if [ "$failed" -eq 0 ]; then
        ok "NSD download complete."
        info "NOTE: Run step 3b (save_nsdsynthetic_stimuli.ipynb) to generate nsdsynthetic_jpg/."
        COMPLETED_DOWNLOADS+=("NSD")
    else
        err "NSD download completed with errors. See messages above."
        FAILED_DOWNLOADS+=("NSD")
        _nsd_manual_instructions "$dest"
    fi
}

_nsd_manual_instructions() {
    local dest="$1"
    info "Manual NSD download commands:"
    info "  # Main stimuli HDF5 (~39 GB):"
    info "  aws s3 cp --no-sign-request \\"
    info "    s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \\"
    info "    $dest/nsd_stimuli.hdf5"
    info "  # NSD synthetic stimuli (~640 MB, needed for step 3b):"
    info "  aws s3 cp --no-sign-request \\"
    info "    s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_stimuli.hdf5 \\"
    info "    $dest/nsdsynthetic_stimuli.hdf5"
    info "  # Stimulus metadata CSV:"
    info "  aws s3 cp --no-sign-request \\"
    info "    s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv \\"
    info "    $dest/nsd_stim_info_merged.csv"
    info "  # COCO 2017 annotations (from cocodataset.org):"
    info "  curl -L -o /tmp/coco_ann.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    info "  unzip /tmp/coco_ann.zip -d $dest/annotations_trainval2017"
    info "  # notshown.tsv (derive from CSV after downloading it):"
    info "  NSD_DEST=$dest uv run --project $PROJECT_ROOT python -c \""
    info "    import os, pandas as pd; dest=os.environ['NSD_DEST']"
    info "    df=pd.read_csv(os.path.join(dest,'nsd_stim_info_merged.csv'))"
    info "    cols=[c for c in df.columns if c.startswith('subject') and c[7:].isdigit()]"
    info "    ids=[i+1 for i in df.index[df[cols].sum(1)==0].tolist()]"
    info "    pd.DataFrame(ids).to_csv(os.path.join(dest,'notshown.tsv'),index=False,header=False)"
    info "    \""
    info "Browse the full NSD S3 bucket at:"
    info "  https://natural-scenes-dataset.s3.amazonaws.com/index.html"
}

# ── main ──────────────────────────────────────────────────────────────────────

check_env

echo -e "${BOLD}MOSAIC Stimulus Download Script${NC}"
echo "DATASETS_ROOT = $DATASETS_ROOT"

download_bold5000
download_bmd
download_god_deeprecon
download_had
download_things
download_nod
download_nsd

# ── summary ───────────────────────────────────────────────────────────────────

echo
echo -e "${BOLD}══════════════════════════════════════════════${NC}"
echo -e "${BOLD} Download Summary${NC}"
echo -e "${BOLD}══════════════════════════════════════════════${NC}"

if [ ${#COMPLETED_DOWNLOADS[@]} -gt 0 ]; then
    echo -e "${GREEN}Automatically downloaded:${NC}"
    for d in "${COMPLETED_DOWNLOADS[@]}"; do echo "  ✓ $d"; done
fi

if [ ${#SKIPPED_DOWNLOADS[@]} -gt 0 ]; then
    echo "Already present (skipped):"
    for d in "${SKIPPED_DOWNLOADS[@]}"; do echo "  - $d"; done
fi

if [ ${#FAILED_DOWNLOADS[@]} -gt 0 ]; then
    echo -e "${RED}Failed (see errors above):${NC}"
    for d in "${FAILED_DOWNLOADS[@]}"; do echo "  ✗ $d"; done
fi

if [ ${#MANUAL_DOWNLOADS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Require manual download (see instructions above):${NC}"
    for d in "${MANUAL_DOWNLOADS[@]}"; do echo "  ! $d"; done
fi

total_manual=${#MANUAL_DOWNLOADS[@]}
total_failed=${#FAILED_DOWNLOADS[@]}

echo
if [ "$total_manual" -eq 0 ] && [ "$total_failed" -eq 0 ]; then
    echo -e "${GREEN}All stimuli are ready.${NC} Proceed to step 3b in the README."
else
    if [ "$total_manual" -gt 0 ]; then
        echo "Complete the manual downloads listed above, then proceed to step 3b."
    fi
    if [ "$total_failed" -gt 0 ]; then
        echo -e "${RED}Fix the failed downloads before proceeding.${NC}"
        exit 1
    fi
fi
