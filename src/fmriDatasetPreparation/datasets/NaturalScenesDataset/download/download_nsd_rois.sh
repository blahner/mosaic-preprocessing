#!/bin/bash

set -e
# Make sure you source your .env file before sourcing this script to access the necessary environment variables
LOCAL_DIR="${DATASETS_ROOT}/NaturalScenesDataset"

# Known session counts for each subject
# sub-01 - 40, sub-02 - 40, sub-03 - 32, sub-04 - 30, sub-05 - 40, sub-06 - 32, sub-07 - 40, sub08 - 30
rois=("floc-faces" "floc-words" "floc-bodies" "floc-places" "Kastner2015" "nsdgeneral" "HCP_MMP1" "MTL" "streams" "thalamus")
for sub in {01..08}; do
    echo "Processing subject $sub ..."
    
    # Create the target directory
    target_dir="$LOCAL_DIR/derivatives/nsddata/ppdata/sub-${sub}/func1pt8mm/roi"
    mkdir -p "$target_dir"
    
    # Set the S3 prefix for this subject
    s3_prefix="s3://natural-scenes-dataset/nsddata/ppdata/subj${sub}/func1pt8mm/roi"

    for roi in "${rois[@]}"; do
        # Construct the full S3 path for this file
        s3_file="$s3_prefix/${roi}.nii.gz"
            
        # Download the file
        if aws s3 cp --no-sign-request "$s3_file" "$target_dir/"; then
            echo "Successfully downloaded"
        else
            echo "Warning: Failed to download subject $sub"
            # Continue despite errors since we're using set -e
        fi
    done
    
    echo "Completed subject $sub"
done

echo "Finished all subjects in the loop"