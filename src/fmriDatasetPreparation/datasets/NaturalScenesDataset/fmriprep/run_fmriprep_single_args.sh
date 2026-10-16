set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
export ROOT="${DATASETS_ROOT}/NaturalScenesDataset"
export OUTPUT_RELPATH=/derivatives
export WORK=${TMP}/tmp/NSD-workdir
export FMRIPREP_VERSION="25.2.5"
echo "${DATASETS_ROOT}" 
mkdir -p ${WORK}
# create the /out bind-mount source ourselves: if docker auto-creates a missing
# bind-mount path it lands root-owned, and the --user container then cannot
# write into it (fmriprep fails with PermissionError on '/out/logs')
mkdir -p "${ROOT}/${OUTPUT_RELPATH}/fmriprep"

# NSD is by far the largest dataset here (400-536 runs/subject vs 10-145
# elsewhere), and sub-07 showed the box sitting mostly idle at 32/8 - load
# averaged well under 10 for most of its 97h. Raised to 48/12 to give each
# subject more of this 64-core machine while still leaving headroom for other
# users. Disk, not CPU, is the limit on running subjects concurrently.
nthreads=48
omp_nthreads=12
docker pull nipreps/fmriprep:${FMRIPREP_VERSION}
# fMRIPrep names its work tree fmriprep_<major>_<minor>_wf/, derived here from
# FMRIPREP_VERSION so the cleanup below can't go stale when the version changes
WF_ROOT="${WORK}/fmriprep_$(echo "${FMRIPREP_VERSION}" | cut -d. -f1,2 | tr . _)_wf"

echo "Starting fMRIPrep for sub-$1"
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
    participant --participant-label $1 \
    --output-space MNI152NLin2009cAsym:res-2 fsaverage anat fsnative \
    --fs-license-file /opt/freesurfer_license/license.txt \
    --cifti-output 91k \
    --bold2anat-dof 6 \
    --no-track-sessions \
    --slice-time-ref 0 \
    --nthreads $nthreads \
    --omp-nthreads $omp_nthreads \
    --stop-on-first-crash \
    -w /work
echo "Deleting the large tmp files from subject $1"
# subject dir is sub_<id>_wf, or sub_<id>_ses_<ses>_wf when processed session-wise
shopt -s nullglob
subj_wf_dirs=("${WF_ROOT}"/sub_${1}_*wf)
shopt -u nullglob
if [ ${#subj_wf_dirs[@]} -eq 0 ]; then
    echo "WARNING: no work dir matching ${WF_ROOT}/sub_${1}_*wf - nothing deleted"
else
    rm -r "${subj_wf_dirs[@]}"
fi

echo "Finished fMRIPrep for the subject"