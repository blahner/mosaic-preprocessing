set -e
# Define root. Source the project's .env file first to access proejct root variable.
LOCAL_DIR="${DATASETS_ROOT}/deeprecon"

mkdir ${LOCAL_DIR}/Nifti
dataset_files=("dataset_description.json"
    "README"
    "CHANGES")

for f in "${dataset_files[@]}"; do
    aws s3 cp --no-sign-request s3://openneuro.org/ds001506/${f} \
    ${LOCAL_DIR}/Nifti/
done

for sub in {01..03}; do
    mkdir ${LOCAL_DIR}/Nifti/sub-${sub}
    aws s3 sync --no-sign-request s3://openneuro.org/ds001506/sub-${sub}/ \
    ${LOCAL_DIR}/Nifti/sub-${sub}
done