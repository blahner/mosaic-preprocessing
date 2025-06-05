set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
LOCAL_DIR="${DATASETS_ROOT}/BOLD5000"

dataset_files=("dataset_description.json"
    "README"
    "CHANGES"
    "participants.tsv"
    "task-5000scenes_bold.json"
    "task-5000scenes_events.json"
    "task-localizer_bold.json"
    "task-localizer_events.json")

mkdir ${LOCAL_DIR}/Nifti
for f in "${dataset_files[@]}"; do
    aws s3 cp --no-sign-request s3://openneuro.org/ds001499/${f} \
    ${LOCAL_DIR}/Nifti/
done

for sub in {1..4}; do
    mkdir ${LOCAL_DIR}/Nifti/sub-CSI${sub}
    aws s3 sync --no-sign-request s3://openneuro.org/ds001499/sub-CSI${sub}/ \
    ${LOCAL_DIR}/Nifti/sub-CSI${sub}
done