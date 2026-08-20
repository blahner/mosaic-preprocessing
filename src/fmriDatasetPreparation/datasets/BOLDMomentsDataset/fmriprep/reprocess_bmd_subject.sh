#!/bin/bash
set -e
# End-to-end single-subject reprocessing pipeline for BOLDMomentsDataset:
# fmriprep -> coreg-determinant QA gate -> GLMsingle (all sessions) ->
# organize_betas -> noiseceiling_compare -> QC movie. One blocking script,
# one background launch, one completion notification - instead of chaining
# separate backgrounded stages by hand across conversation turns and relying
# on each one being noticed before the next is kicked off.
#
# Resumable: a multi-hour background job silently stopping partway through
# (killed process, host reset, session boundary - whatever the cause) is a
# real, observed failure mode for jobs this long, and re-running the whole
# thing from scratch every time is wasteful and risky. Each stage checks for
# its own completion marker first and skips itself if already done, so
# re-running this script after an interruption resumes instead of restarting.
#
# Never touches versionC (or overwrites a completed run of a version):
# everything writes under a new/resumable derivatives/<version> tree, and
# versionC specifically is always refused outright.
#
# Usage:
#   reprocess_bmd_subject.sh <subject_num_2digit e.g. 05> <new_version e.g. versionD> [extra fmriprep args...]
#
# Example (the sub-05 dof=6 test this script was extracted from):
#   reprocess_bmd_subject.sh 05 versionD --bold2t1w-dof 6
#
# Requires: DATASETS_ROOT, PROJECT_ROOT, FREESURFER_HOME set (source .env first).
# Requires: docker; the mosaic-preprocessing conda env for everything after
# fmriprep (see CLAUDE.md "Environment setup" for why this can't just be
# `conda run`/bare `python3`).

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <subject_num_2digit> <new_version> [extra fmriprep args...]"
    exit 1
fi
SUBJ=$1
VERSION=$2
shift 2
EXTRA_FMRIPREP_ARGS=("$@")

MPY=/data/vision/oliva/blahner/anaconda3/envs/mosaic-preprocessing/bin
export ROOT="${DATASETS_ROOT}/BOLDMomentsDataset"
export OUTPUT_RELPATH="/derivatives/${VERSION}"
export WORK="/tmp/mosaic-fmriprep-work/BMD-sub${SUBJ}-${VERSION}"
export FMRIPREP_VERSION="23.2.0"
QA_DIR="${PROJECT_ROOT}/src/fmriDatasetPreparation/qa"
GLM_DIR="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/BOLDMomentsDataset/GLM"
VAL_DIR="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/BOLDMomentsDataset/validation"
VIZ_DIR="${PROJECT_ROOT}/src/fmriDatasetPreparation/visualizations"

if [ "${VERSION}" = "versionC" ]; then
    echo "REFUSING to run: versionC is the dataset's primary derivatives version and must never be targeted by this script."
    exit 1
fi

FMRIPREP_DONE_MARKER="${ROOT}${OUTPUT_RELPATH}/fmriprep/sub-${SUBJ}.html"
if [ -f "${FMRIPREP_DONE_MARKER}" ]; then
    echo "### [1/5] fMRIPrep: sub-${SUBJ} -> ${VERSION} already completed (found ${FMRIPREP_DONE_MARKER}) - skipping"
else
    echo "### [1/5] fMRIPrep: sub-${SUBJ} -> ${VERSION} (extra args: ${EXTRA_FMRIPREP_ARGS[*]})"
    mkdir -p "${WORK}"
    mkdir -p "${ROOT}${OUTPUT_RELPATH}/fmriprep"   # pre-create: letting docker auto-create this bind-mount source hits an NFS permission error on this filesystem
    docker run \
        --user "$(id -u):$(id -g)" \
        --rm \
        -v "$ROOT/Nifti":/data:ro \
        -v "${ROOT}${OUTPUT_RELPATH}/fmriprep":/out \
        -v "$WORK":/work \
        -v "$FREESURFER_HOME/license.txt":/opt/freesurfer_license/license.txt \
        nipreps/fmriprep:${FMRIPREP_VERSION} \
        /data /out \
        --skip_bids_validation \
        participant --participant-label "${SUBJ}" \
        --output-space MNI152NLin2009cAsym:res-2 \
        --fs-license-file /opt/freesurfer_license/license.txt \
        --cifti-output 91k \
        --slice-time-ref 0 \
        --nthreads 8 \
        --n-cpus 16 \
        --stop-on-first-crash \
        "${EXTRA_FMRIPREP_ARGS[@]}" \
        -w /work
    if [ ! -f "${FMRIPREP_DONE_MARKER}" ]; then
        echo "fMRIPrep did not produce ${FMRIPREP_DONE_MARKER} - treating as failed, stopping."
        exit 1
    fi
fi

echo "### [2/5] QA gate: coregistration determinant check"
PYTHONPATH="${PROJECT_ROOT}" "${MPY}/python3" "${QA_DIR}/coreg_determinant_check.py" \
    --fmriprep-dir "${ROOT}${OUTPUT_RELPATH}/fmriprep" \
    --subs "sub-${SUBJ}" \
    --out-csv "${ROOT}${OUTPUT_RELPATH}/coreg_determinant_report_sub-${SUBJ}.csv"
# Deliberately not gating the pipeline on this (`set -e` would stop the whole
# script on any flagged run) - flagged runs are a strong prior worth a human
# look, not an automatic verdict. Report is written either way; read it.

echo "### [3/5] GLMsingle: sub-${SUBJ}, sessions 2-5 -> ${VERSION}"
for ses in 2 3 4 5; do
    GLM_DONE_MARKER="${ROOT}${OUTPUT_RELPATH}/GLM/sub-${SUBJ}/ses-0${ses}/TYPED_FITHRF_GLMDENOISE_RR.npy"
    if [ -f "${GLM_DONE_MARKER}" ]; then
        echo "  -- session ${ses} already completed (found ${GLM_DONE_MARKER}) - skipping --"
        continue
    fi
    echo "  -- session ${ses} --"
    PYTHONPATH="${PROJECT_ROOT}" "${MPY}/python3" "${GLM_DIR}/glmsingle_bmd.py" \
        -s "${SUBJ}" -i "${ses}" --version "${VERSION}" -v
done

echo "### [4/5] organize_betas + noise ceiling: sub-${SUBJ} -> ${VERSION}"
PYTHONPATH="${PROJECT_ROOT}" "${MPY}/python3" "${GLM_DIR}/organize_betas.py" \
    -s "${SUBJ}" --version "${VERSION}" -v
PYTHONPATH="${PROJECT_ROOT}" "${MPY}/python3" "${VAL_DIR}/noiseceiling_compare.py" \
    -s "${SUBJ}" --version "${VERSION}"

echo "### [5/5] QC movie: sub-${SUBJ} -> ${VERSION}"
PYTHONPATH="${PROJECT_ROOT}" "${MPY}/python3" "${VIZ_DIR}/bmd_fmriprep_qc_movie.py" \
    --subs "sub-${SUBJ}" --version "${VERSION}"

echo "### DONE: sub-${SUBJ} fully reprocessed under derivatives/${VERSION}"
