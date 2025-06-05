set -e
LOCAL_DIR=${DATASETS_ROOT}/THINGS_fmri

dataset_files=("dataset_description.json"
    "participants.json"
    "participants.tsv"
    "README"
    "CHANGES"
    "task-things_events.json"
)
mkdir -p ${LOCAL_DIR}/Nifti
for f in "${dataset_files[@]}"; do
    aws s3 cp --no-sign-request s3://openneuro.org/ds004192/${f} \
    ${LOCAL_DIR}/Nifti/
done

for sub in {01..03}; do
    mkdir ${LOCAL_DIR}/Nifti/sub-${sub}
    aws s3 sync --no-sign-request s3://openneuro.org/ds004192/sub-${sub}/ \
    ${LOCAL_DIR}/Nifti/sub-${sub}
done