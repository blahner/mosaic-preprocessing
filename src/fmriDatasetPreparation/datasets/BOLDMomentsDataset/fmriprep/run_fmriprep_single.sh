set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
export ROOT="${DATASETS_ROOT}/BOLDMomentsDataset"
export OUTPUT_RELPATH=/derivatives/versionC
export WORK=${TMP}/tmp/BMD-workdir
export FMRIPREP_VERSION="23.2.0"
echo "${DATASETS_ROOT}" 
mkdir -p ${WORK}
mkdir -p "${ROOT}/${OUTPUT_RELPATH}"

nthreads=8
ncpus=16
docker pull nipreps/fmriprep:${FMRIPREP_VERSION}

for subj in {01..10}; do
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
    --output-space MNI152NLin2009cAsym:res-2 \
    --fs-license-file /opt/freesurfer_license/license.txt \
    --cifti-output 91k \
    --bold2t1w-dof 12 \
    --slice-time-ref 0 \
    --nthreads $nthreads \
    --n-cpus $ncpus \
    --stop-on-first-crash \
    -w /work
    echo "Deleting the large tmp files from subject ${subj}"
    rm -r ${WORK}/fmriprep_23_2_wf/sub_${subj}_wf/
done
echo "Finished fMRIPrep for all subjects in the loop"
