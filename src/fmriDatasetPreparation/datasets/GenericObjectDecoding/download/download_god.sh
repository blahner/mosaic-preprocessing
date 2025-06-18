set -e
LOCAL_DIR=${DATASETS_ROOT}/GenericObjectDecoding

dataset_files=("dataset_description.json"
    "README"
    "CHANGES"
    "T1w.json"
    "inplaneT2.json"
    "task-imagery_bold.json"
    "task-perception_bold.json")

mdkir ${LOCAL_DIR}/Nifti
for f in "${dataset_files[@]}"; do
    aws s3 cp --no-sign-request s3://openneuro.org/ds001246/${f} \
    ${LOCAL_DIR}/Nifti/
done

for sub in {01..05}; do
    mkdir ${LOCAL_DIR}/Nifti/sub-${sub}
    aws s3 sync --no-sign-request s3://openneuro.org/ds001246/sub-${sub}/ \
    ${LOCAL_DIR}/Nifti/sub-${sub}
done