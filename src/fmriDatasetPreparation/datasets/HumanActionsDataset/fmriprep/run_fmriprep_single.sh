set -e
export NII=/data/vision/oliva/scratch/datasets/HumanActionsDataset
export TMP=/data/vision/oliva/scratch/blahner
export WORK=$TMP/tmp/HAD-workdir
export FMRIPREP_VERSION="23.2.0"
export USERNAME=blahner

su $USERNAME -c "mkdir -p $NII/derivatives"
su $USERNAME -c "mkdir -p $WORK"

nthreads=4
ncpus=16
docker pull nipreps/fmriprep:${FMRIPREP_VERSION}

for subj in {02..30}; do
    echo "Starting fMRIPrep for sub-${subj}"
    docker run \
    --user 24591:20681 \
    -it --rm \
    -v $NII/Nifti:/data:ro \
    -v $NII/derivatives:/out \
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
    rm -r $WORK/fmriprep_23_2_wf/sub_${subj}_wf/
done
echo "Finished fMRIPrep for all subjects in the loop"
