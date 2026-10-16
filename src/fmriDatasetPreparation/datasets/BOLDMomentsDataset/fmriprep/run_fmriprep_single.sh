set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
export ROOT="${DATASETS_ROOT}/BOLDMomentsDataset"
export OUTPUT_RELPATH=/derivatives/versionC
export WORK=${TMP}/tmp/BMD-workdir
export FMRIPREP_VERSION="25.2.5"
echo "${DATASETS_ROOT}" 
mkdir -p ${WORK}
# create the /out bind-mount source ourselves: if docker auto-creates a missing
# bind-mount path it lands root-owned, and the --user container then cannot
# write into it (fmriprep fails with PermissionError on '/out/logs')
mkdir -p "${ROOT}/${OUTPUT_RELPATH}/fmriprep"

# total threads = at most half of this shared machine's 64 cores; one subject at a time
nthreads=32
omp_nthreads=8
docker pull nipreps/fmriprep:${FMRIPREP_VERSION}
# fMRIPrep names its work tree fmriprep_<major>_<minor>_wf/, derived here from
# FMRIPREP_VERSION so the cleanup below can't go stale when the version changes
WF_ROOT="${WORK}/fmriprep_$(echo "${FMRIPREP_VERSION}" | cut -d. -f1,2 | tr . _)_wf"

FAILED_SUBJECTS=()
for subj in {01..10}; do
    # Resumable: lets the loop be re-run after an interruption (host reboot, a
    # crashed subject) without redoing finished subjects. The done-marker is
    # actual CIFTI output, NOT the HTML report: fMRIPrep writes a report even
    # when a run crashes (BMD sub-03 2026-10-09 had full reports but zero
    # derivatives after a recon-all segfault), so keying on the report would
    # make a failed subject look finished and be skipped forever.
    if [ -n "$(find "${ROOT}/${OUTPUT_RELPATH}/fmriprep/sub-${subj}" -name '*.dtseries.nii' -print -quit 2>/dev/null)" ]; then
        echo "Skipping sub-${subj}: already has fmriprep derivatives"
        continue
    fi
    # a crashed run leaves FreeSurfer IsRunning locks that block recon-all on resume
    find "${ROOT}/${OUTPUT_RELPATH}/fmriprep/sourcedata/freesurfer/sub-${subj}/scripts" \
        -name 'IsRunning*' -delete 2>/dev/null || true
    echo "Starting fMRIPrep for sub-${subj}"
    docker run \
    --user $(id -u):$(id -g) \
    -it --rm \
    -v $ROOT/Nifti:/data:ro \
    -v "${ROOT}/${OUTPUT_RELPATH}/fmriprep":/out \
    -v $WORK:/work \
    -v $FREESURFER_HOME/license.txt:/opt/freesurfer_license/license.txt \
    \
    nipreps/fmriprep:${FMRIPREP_VERSION} \
    /data /out \
    --skip_bids_validation \
    participant --participant-label ${subj} \
    --output-space MNI152NLin2009cAsym:res-2 fsaverage anat fsnative \
    --fs-license-file /opt/freesurfer_license/license.txt \
    --cifti-output 91k \
    --bold2anat-dof 6 \
    --no-track-sessions \
    --slice-time-ref 0 \
    --nthreads $nthreads \
    --omp-nthreads $omp_nthreads \
    --stop-on-first-crash \
    -w /work || { echo "WARNING: sub-${subj} FAILED - continuing to next subject"; FAILED_SUBJECTS+=("${subj}"); }
    echo "Deleting the large tmp files from subject ${subj}"
    # subject dir is sub_<id>_wf, or sub_<id>_ses_<ses>_wf when processed session-wise
    shopt -s nullglob
    subj_wf_dirs=("${WF_ROOT}"/sub_${subj}_*wf)
    shopt -u nullglob
    if [ ${#subj_wf_dirs[@]} -eq 0 ]; then
        echo "WARNING: no work dir matching ${WF_ROOT}/sub_${subj}_*wf - nothing deleted"
    else
        rm -r "${subj_wf_dirs[@]}"
    fi
done
echo "Finished fMRIPrep for all subjects in the loop"
if [ ${#FAILED_SUBJECTS[@]} -gt 0 ]; then echo "FAILED SUBJECTS: ${FAILED_SUBJECTS[*]}"; fi
