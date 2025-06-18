set -e
LOCAL_DIR=${DATASETS_DIR}/NaturalObjectDataset

dataset_files=("dataset_description.json"
    "participants.json"
    "participants.tsv"
    "README"
    "CHANGES"
    "task-coco_bold.json"
    "task-imagenet_events.json"
    "task-imagenet_bold.json"
    "task-floc_bold.json"
    "task-prf_bold.json")

mkdir ${LOCAL_DIR}/Nifti
for f in "${dataset_files[@]}"; do
    aws s3 cp --no-sign-request s3://openneuro.org/ds004496/${f} \
    ${LOCAL_DIR}/Nifti/
done

for sub in {01..30}; do
    mkdir ${LOCAL_DIR}/Nifti/sub-${sub}
    aws s3 sync --no-sign-request s3://openneuro.org/ds004496/sub-${sub}/ \
    ${LOCAL_DIR}/Nifti/sub-${sub}
done