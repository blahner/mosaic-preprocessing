#!/bin/bash
set -e
# One-off diagnostic rerun of fMRIPrep for BMD sub-05 with --bold2t1w-dof 6
# (rigid body, fMRIPrep's own documented default) instead of the project's
# usual --bold2t1w-dof 12, to test whether that setting explains the
# brain-scaling mismatches seen in the sub-05 QC movie (see
# src/fmriDatasetPreparation/qa/README.md for the full pattern).
#
# Mirrors run_fmriprep_single.sh exactly except: single subject (05),
# --bold2t1w-dof 6, and a new output version (versionD) so the existing
# versionC derivatives are untouched.
export ROOT="${DATASETS_ROOT}/BOLDMomentsDataset"
export OUTPUT_RELPATH=/derivatives/versionD
export WORK=/tmp/mosaic-fmriprep-work/BMD-sub05-dof6
export FMRIPREP_VERSION="23.2.0"
echo "${DATASETS_ROOT}"
mkdir -p ${WORK}
# Pre-create the bind-mount output dir ourselves: letting `docker run` create
# it on first mount hit an NFS permission error on this filesystem.
mkdir -p "${ROOT}/${OUTPUT_RELPATH}/fmriprep"

nthreads=8
ncpus=16

subj=05
echo "Starting fMRIPrep (dof=6 test) for sub-${subj}"
docker run \
--user $(id -u):$(id -g) \
--rm \
-v $ROOT/Nifti:/data:ro \
-v "${ROOT}/${OUTPUT_RELPATH}/fmriprep":/out \
-v $WORK:/work \
-v $FREESURFER_HOME/license.txt:/opt/freesurfer_license/license.txt \
\
nipreps/fmriprep:${FMRIPREP_VERSION} \
/data /out \
--skip_bids_validation \
participant --participant-label ${subj} \
--output-space MNI152NLin2009cAsym:res-2 \
--fs-license-file /opt/freesurfer_license/license.txt \
--cifti-output 91k \
--bold2t1w-dof 6 \
--slice-time-ref 0 \
--nthreads $nthreads \
--n-cpus $ncpus \
--stop-on-first-crash \
-w /work
echo "Finished fMRIPrep (dof=6 test) for sub-${subj}"
