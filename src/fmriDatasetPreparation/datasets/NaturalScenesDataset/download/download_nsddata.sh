set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
LOCAL_DIR="${DATASETS_ROOT}/NaturalScenesDataset"

dataset_files=("dataset_description.json"
    "participants.json"
    "participants.tsv"
    "README"
    "recording-cardiac_physio.json"
    "recording-respiratory_physio.json"
    "task-floc_bold.json"
    "task-floc_events.json"
    "task-nsdcore_bold.json"
    "task-nsdcore_events.json"
    "task-prfbar_bold.json"
    "task-prfwedge_bold.json"
    "task-rest_bold.json")

for f in "${dataset_files[@]}"; do
    aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata_rawdata/${f} \
    $LOCAL_DIR
done
# sub-01 - 40, sub-02 - 40, sub-03 - 32, sub-04 - 30, sub-05 - 40, sub-06 - 32, sub-07 - 40, sub08 - 30

session_folders=("nsddiffusion"
    "prffloc"
    "nsdimagery"
    "nsdsynthetic")

for sub in {01..08}; do
    mkdir -p $LOCAL_DIR/Nifti/sub-${sub}
    for sf in "${session_folders[@]}"; do
        mkdir -p $LOCAL_DIR/Nifti/sub-${sub}/ses-${sf}
        aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata_rawdata/sub-${sub}/ses-${sf}/ \
        $LOCAL_DIR/Nifti/sub-${sub}/ses-${sf} --recursive
    done
done
echo "Finished all subjects in the loop"