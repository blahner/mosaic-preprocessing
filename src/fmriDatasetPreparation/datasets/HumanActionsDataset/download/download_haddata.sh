set -e
# downloading HAD using the "ds004488-1.1.1_noderivatives.sh" script worked best because some updates to the dataset weren't reflected in the s3 bucket

LOCAL_DIR=${DATASETS_ROOT}/HumanActionsDataset/
echo $LOCAL_DIR
aws s3 sync --no-sign-request s3://openneuro.org/ds004488 ${LOCAL_DIR}

#dataset_files=("dataset_description.json"
#    "participants.json"
#    "participants.tsv"
#    "README"
#    "CHANGES"
#    "task-action_bold.json"
#    "task-action_events.json")

#for f in "${dataset_files[@]}"; do
#    echo ${f}
#    aws s3 sync --no-sign-request s3://openneuro.org/ds004488/${f} \
#    ${LOCAL_DIR}/Nifti/
#done

#for sub in {01..30}; do
#    echo ${sub}
#    mkdir -p ${LOCAL_DIR}/Nifti/sub-${sub}
#    aws s3 sync --no-sign-request s3://openneuro.org/ds004488/sub-${sub}/ \
#    ${LOCAL_DIR}/Nifti/sub-${sub}
#done