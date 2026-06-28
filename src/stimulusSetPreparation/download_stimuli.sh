#!/usr/bin/env bash
#
# download_stimuli.sh — Download all MOSAIC stimulus sets
#
# Usage:
#   source .env && bash src/stimulusSetPreparation/download_stimuli.sh
#
# Prerequisites:
#   - DATASETS_ROOT must be set (define it in your .env file)
#   - AWS CLI installed and configured for unauthenticated S3 access
#     (used for NOD, HAD, and NSD)
#
# What this script does:
#   - Automatically downloads stimuli that are hosted on public S3 (NOD, HAD, NSD)
#   - Prints step-by-step manual instructions for datasets that require
#     account registration, browser downloads, or custom tooling (BOLD5000,
#     BMD, GOD/Deeprecon, THINGS)
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
    if ! command -v aws &>/dev/null; then
        warn "AWS CLI not found. Datasets that require S3 (NOD, HAD, NSD) will be skipped."
        NO_AWS=1
    fi
}

# ── BOLD5000 ──────────────────────────────────────────────────────────────────
#
# Expected layout after download:
#   $DATASETS_ROOT/BOLD5000/derivatives/stimuli_metadata/Scene_Stimuli/
#     Presented_Stimuli/COCO/*.jpg       (~2000 images)
#     Presented_Stimuli/ImageNet/*.JPEG  (~1916 images)
#     Presented_Stimuli/Scene/*.jpg      (~1000 images)

download_bold5000() {
    section "BOLD5000"
    local dest="$DATASETS_ROOT/BOLD5000/derivatives/stimuli_metadata/Scene_Stimuli/Presented_Stimuli"
    if has_files "$dest" 100; then
        ok "BOLD5000 stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("BOLD5000")
        return
    fi
    warn "BOLD5000 stimuli require manual download."
    info "1. Go to: https://bold5000-dataset.github.io/website/download.html"
    info "2. Download 'Scene_Stimuli.zip' from the Stimuli section."
    info "3. Extract it so the following path exists:"
    info "     $dest/COCO/*.jpg"
    info "     $dest/ImageNet/*.JPEG"
    info "     $dest/Scene/*.jpg"
    MANUAL_DOWNLOADS+=("BOLD5000")
}

# ── BOLD Moments Dataset (BMD) ────────────────────────────────────────────────
#
# Expected layout after download + step 3b preprocessing:
#   $DATASETS_ROOT/BOLDMomentsDataset/derivatives/stimuli_metadata/
#     annotations.json
#     mp4_h264/*/          (1102 mp4 videos)
#     frames_middle/*.jpg  (created by step 3b: extract_frames_bmd.py)
#     frames/*/            (created by step 3b: extract_frames_bmd.py)

download_bmd() {
    section "BOLD Moments Dataset (BMD)"
    local dest="$DATASETS_ROOT/BOLDMomentsDataset/derivatives/stimuli_metadata"
    if has_files "$dest/mp4_h264" 100; then
        ok "BMD stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("BMD")
        return
    fi
    warn "BMD stimuli require manual download."
    info "Follow the download instructions at:"
    info "  https://github.com/blahner/BOLDMomentsDataset"
    info "Download the mp4 videos and place them at:"
    info "  $dest/mp4_h264/"
    info "Also download annotations.json to:"
    info "  $dest/annotations.json"
    MANUAL_DOWNLOADS+=("BMD")
}

# ── Generic Object Decoding + Deeprecon ───────────────────────────────────────
#
# GOD and Deeprecon share the same task stimuli (ImageNet images).
# Expected layout:
#   $DATASETS_ROOT/GenericObjectDecoding/derivatives/stimuli_metadata/images/
#     test/*.JPEG      (50 images)
#     training/*.JPEG  (1200 images)
#   $DATASETS_ROOT/deeprecon/derivatives/stimuli_metadata/images/
#     ArtificialImage/*.tiff  (40 images)
#     LetterImage/*.tif       (10 images)

download_god_deeprecon() {
    section "Generic Object Decoding / Deeprecon"
    local dest_god="$DATASETS_ROOT/GenericObjectDecoding/derivatives/stimuli_metadata/images"
    local dest_dr="$DATASETS_ROOT/deeprecon/derivatives/stimuli_metadata/images"

    if has_files "$dest_god/test" 10 && has_files "$dest_god/training" 100; then
        ok "GOD task stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("GOD-task-stimuli")
    else
        warn "GOD/Deeprecon task stimuli (ImageNet images) require manual download."
        info "Follow the download instructions at:"
        info "  https://github.com/KamitaniLab/GenericObjectDecoding"
        info "  (or) https://github.com/KamitaniLab/DeepImageReconstruction"
        info "Place test images at:     $dest_god/test/"
        info "Place training images at: $dest_god/training/"
        MANUAL_DOWNLOADS+=("GOD-task-stimuli")
    fi

    if has_files "$dest_dr/ArtificialImage" 1 && has_files "$dest_dr/LetterImage" 1; then
        ok "Deeprecon artificial/letter stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("Deeprecon-artificial-stimuli")
    else
        warn "Deeprecon artificial/letter stimuli require manual download."
        info "Download from: https://github.com/KamitaniLab/DeepImageReconstruction"
        info "Place .tiff files at: $dest_dr/ArtificialImage/"
        info "Place .tif  files at: $dest_dr/LetterImage/"
        MANUAL_DOWNLOADS+=("Deeprecon-artificial-stimuli")
    fi
}

# ── Human Actions Dataset (HAD) ───────────────────────────────────────────────
#
# Expected layout after download + step 3b preprocessing:
#   $DATASETS_ROOT/HumanActionsDataset/Nifti/stimuli/
#     <category>/*.mp4     (180 categories × 120 videos each)
#   derivatives/stimuli_metadata/frames_middle/*.jpg  (created by step 3b)
#   derivatives/stimuli_metadata/frames/*/            (created by step 3b)

download_had() {
    section "Human Actions Dataset (HAD)"
    local dest="$DATASETS_ROOT/HumanActionsDataset/Nifti/stimuli"
    if has_files "$dest" 100; then
        ok "HAD stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("HAD")
        return
    fi
    if [ -n "${NO_AWS}" ]; then
        warn "AWS CLI not available — skipping automated HAD download."
        info "Install the AWS CLI, then re-run this script, or download manually:"
        info "  aws s3 sync --no-sign-request s3://openneuro.org/ds004488/stimuli/ \\"
        info "    $dest/"
        MANUAL_DOWNLOADS+=("HAD")
        return
    fi
    info "Downloading HAD stimuli from OpenNeuro (~180 categories of mp4 videos)..."
    mkdir -p "$dest"
    if aws s3 sync --no-sign-request s3://openneuro.org/ds004488/stimuli/ "$dest/"; then
        ok "HAD download complete."
        COMPLETED_DOWNLOADS+=("HAD")
    else
        err "HAD download failed. Check your network connection and retry."
        FAILED_DOWNLOADS+=("HAD")
    fi
}

# ── THINGS ────────────────────────────────────────────────────────────────────
#
# Expected layout after download + identify_experimental_stimuli.ipynb:
#   $DATASETS_ROOT/THINGS_fmri/derivatives/stimuli_metadata/experimental_images/
#     *.jpg  (8740 images)

download_things() {
    section "THINGS"
    local dest="$DATASETS_ROOT/THINGS_fmri/derivatives/stimuli_metadata/experimental_images"
    if has_files "$dest" 100; then
        ok "THINGS stimuli already present — skipping."
        SKIPPED_DOWNLOADS+=("THINGS")
        return
    fi
    warn "THINGS stimuli require manual download from OSF."
    info "Option A — download the full THINGS image database (~146 GB):"
    info "  1. Go to https://osf.io/jum2f/ and download all THINGS images."
    info "  2. Run the notebook to extract the 8740 experimental images:"
    info "       src/fmriDatasetPreparation/datasets/THINGS_fmri/download/identify_experimental_stimuli.ipynb"
    info "  3. Final images will be placed at: $dest/"
    info ""
    info "Option B — download only the 8740 experimental images:"
    info "  Use the filename list to selectively download:"
    info "    src/fmriDatasetPreparation/datasets/THINGS_fmri/download/THINGS_fmri_filenames.txt"
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
        python3 - <<'PYEOF'
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
print(f"  Wrote {len(not_shown_1indexed)} not-shown image IDs to {out_path}")
PYEOF
        NSD_DEST="$dest" python3 - <<'PYEOF' 2>&1 | while IFS= read -r line; do info "    $line"; done
import os, pandas as pd
dest = os.environ.get("NSD_DEST")
csv_path = os.path.join(dest, "nsd_stim_info_merged.csv")
df = pd.read_csv(csv_path)
subj_cols = [c for c in df.columns if c.startswith("subject") and c[7:].isdigit()]
not_shown = df.index[df[subj_cols].sum(axis=1) == 0].tolist()
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
    info "  NSD_DEST=$dest python3 -c \""
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
