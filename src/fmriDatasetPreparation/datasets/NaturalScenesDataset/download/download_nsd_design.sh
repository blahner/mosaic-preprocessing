set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
LOCAL_DIR="${DATASETS_ROOT}/NaturalScenesDataset"

session_folders=("nsdsynthetic")
for sub in {01..08}; do
    mkdir -p $LOCAL_DIR/derivatives/design/sub-${sub}
    aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata_timeseries/ppdata/subj${sub}/func1mm/design/ \
    $LOCAL_DIR/derivatives/design/sub-${sub}/ --recursive
done
echo "Finished all subjects in the loop"