set -e
LOCAL_DIR=${DATASETS_ROOT}/NFED

dataset_files=("dataset_description.json"
    "participants.json"
    "participants.tsv"
    "README"
    "CHANGES"
    "task-face_bold.json"
    "task-floc_bold.json"
    "task-prf_bold.json"
)
mkdir -p ${LOCAL_DIR}/Nifti
for f in "${dataset_files[@]}"; do
    aws s3 cp --no-sign-request s3://openneuro.org/ds005047/${f} \
    ${LOCAL_DIR}/Nifti/
done

for sub in {01..05}; do
    mkdir ${LOCAL_DIR}/Nifti/sub-${sub}
    aws s3 sync --no-sign-request s3://openneuro.org/ds005047/sub-${sub}/ \
    ${LOCAL_DIR}/Nifti/sub-${sub}
done

mkdir ${LOCAL_DIR}/Nifti/stimuli
aws s3 sync --no-sign-request s3://openneuro.org/ds005047/stimuli/ \
${LOCAL_DIR}/Nifti/stimuli