set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
LOCAL_DIR="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/NaturalScenesDataset/validation"
echo "LOCAL_DIR: ${LOCAL_DIR}"

for sub in 01; do
    mkdir -p $LOCAL_DIR/output/nsd_freesurfer_original/sub-${sub}
    aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata_other/freesurferoriginals/subj${sub}_original/ \
    $LOCAL_DIR/output/nsd_freesurfer_original/sub-${sub}/ --recursive 
done
echo "Finished all subjects in the loop"