#!/bin/bash
# Batch wrapper around reprocess_bmd_subject.sh: runs the full per-subject
# pipeline (fmriprep -> coreg QA gate -> GLMsingle -> organize_betas ->
# noiseceiling_compare -> QC movie) for a LIST of subjects, a few at a time,
# in the background. See ../../../qa/README.md section 3 for the full
# writeup of when/why this exists and how to check on it - that's the
# canonical reference for a fresh session with no memory of this
# conversation to reconstruct what's running and why.
#
# Origin: BMD sub-05's QC movie showed a brain-scaling artifact, traced to
# --bold2t1w-dof 12 (full affine BOLD->T1w registration, used project-wide)
# occasionally letting an individual run's coregistration converge to a
# spurious scale instead of pure rigid motion. Scanning all 10 BMD subjects'
# EXISTING versionC fmriprep output with coreg_determinant_check.py found
# 5 subjects with at least one flagged run - sub-01 (1/62), sub-02 (1/62),
# sub-05 (11/62), sub-08 (4/62), sub-09 (7/62) - and 5 subjects completely
# clean even under dof=12 (sub-03, 04, 06, 07, 10). Started 2026-08-20,
# reprocessing only the 5 affected subjects under a new derivatives/versionD
# with --bold2t1w-dof 6, to bound disk usage (~54GB/subject output, and the
# shared dataset filesystem was at 96% full / 763GB free when this started -
# reprocessing all 10 subjects instead of the 5 affected ones was considered
# and deliberately rejected for that reason, not because it wouldn't also be
# beneficial for the other 5).
#
# Usage:
#   reprocess_all_bmd_subjects.sh <new_version> <max_parallel> "<space-separated 2-digit subject list>" [extra fmriprep args...]
#
# The run this script was written for:
#   reprocess_all_bmd_subjects.sh versionD 2 "01 02 05 08 09" --bold2t1w-dof 6
#
# Each subject's reprocess_bmd_subject.sh call is itself resumable (skips
# fmriprep/each GLMsingle session if already done under that version) - see
# that script's header. That means THIS script is also safe to just re-run
# after any interruption: subjects/stages already completed are skipped, not
# redone. It does NOT retry a subject whose reprocess_bmd_subject.sh exited
# non-zero within the same invocation - check that subject's log and re-run
# this script to pick it back up.
#
# Requires: DATASETS_ROOT, PROJECT_ROOT, FREESURFER_HOME set (source .env
# first, with `set -a` so they're exported - see CLAUDE.md).

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <new_version> <max_parallel> \"<space-separated subject list>\" [extra fmriprep args...]"
    echo "e.g.:  $0 versionD 2 \"01 02 05 08 09\" --bold2t1w-dof 6"
    exit 1
fi
VERSION=$1
MAX_PARALLEL=$2
SUBJECTS=$3
shift 3
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${DATASETS_ROOT}/BOLDMomentsDataset/derivatives/${VERSION}/reprocess_logs"
mkdir -p "${LOG_DIR}"

echo "Batch reprocessing sub-{${SUBJECTS}} -> ${VERSION}, max ${MAX_PARALLEL} concurrent, extra fmriprep args: ${EXTRA_ARGS[*]}"
echo "Per-subject logs: ${LOG_DIR}/sub-XX.log"

# Throttle to MAX_PARALLEL concurrent subjects, then wait for everything.
# Deliberately does NOT try to track per-PID exit status through the `wait -n`
# throttle below (bash reaps a job's status the first time any `wait` variant
# observes it finish, so a later `wait $pid` for that same job is unreliable)
# - instead, success is read back from each subject's own log after
# everything finishes, via the "### DONE:" marker reprocess_bmd_subject.sh
# writes on success. Simpler and doesn't depend on bash job-control edge cases.
for subj in ${SUBJECTS}; do
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
        wait -n
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching sub-${subj}"
    "${SCRIPT_DIR}/reprocess_bmd_subject.sh" "${subj}" "${VERSION}" "${EXTRA_ARGS[@]}" \
        > "${LOG_DIR}/sub-${subj}.log" 2>&1 &
done
wait

echo ""
FAILED=()
for subj in ${SUBJECTS}; do
    if grep -q "^### DONE:" "${LOG_DIR}/sub-${subj}.log" 2>/dev/null; then
        echo "sub-${subj}: DONE"
    else
        echo "sub-${subj}: FAILED or incomplete (check ${LOG_DIR}/sub-${subj}.log)"
        FAILED+=("${subj}")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "### ALL SUBJECTS (${SUBJECTS}) DONE under derivatives/${VERSION}"
else
    echo "### DONE with failures/incomplete: ${FAILED[*]}"
    echo "### Check the logs above, then re-run this script (it will skip completed subjects/stages, safe to re-run)."
fi
