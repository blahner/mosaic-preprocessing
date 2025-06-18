set -e
LOCAL_DIR=${DATASETS_DIR}/NaturalObjectDataset

mkdir ${LOCAL_DIR}/Nifti/stimuli
aws s3 sync --no-sign-request s3://openneuro.org/ds004496/stimuli/ ${LOCAL_DIR}/Nifti/stimuli
