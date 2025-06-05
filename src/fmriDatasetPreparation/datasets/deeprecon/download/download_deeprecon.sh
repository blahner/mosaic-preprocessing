set -e
LOCAL_DIR=/data/vision/oliva/scratch/datasets/deeprecon

dataset_files=("dataset_description.json"
    "README"
    "CHANGES")

#for f in "${dataset_files[@]}"; do
#    aws s3 cp --no-sign-request s3://openneuro.org/ds001506/${f} \
#    ${LOCAL_DIR}/Nifti/
#done

for sub in {01..03}; do
    mkdir ${LOCAL_DIR}/Nifti/sub-${sub}
    aws s3 sync --no-sign-request s3://openneuro.org/ds001506/sub-${sub}/ \
    ${LOCAL_DIR}/Nifti/sub-${sub}
done