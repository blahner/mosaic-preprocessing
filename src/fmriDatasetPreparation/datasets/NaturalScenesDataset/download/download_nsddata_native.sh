#!/bin/bash

set -e
# Make sure you source your .env file before sourcing this script to access the necessary environment variables
LOCAL_DIR="${DATASETS_ROOT}/NaturalScenesDataset"

# Known session counts for each subject
# sub-01 - 40, sub-02 - 40, sub-03 - 32, sub-04 - 30, sub-05 - 40, sub-06 - 32, sub-07 - 40, sub08 - 30
declare -A sessions
sessions=([01]=40 [02]=40 [03]=32 [04]=30 [05]=40 [06]=32 [07]=40 [08]=30)

for sub in {01..08}; do
    echo "Processing subject $sub with ${sessions[$sub]} sessions..."
    
    # Create the target directory
    target_dir="$LOCAL_DIR/derivatives/nsddata_betas/ppdata/sub-${sub}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    mkdir -p "$target_dir"
    
    # Set the S3 prefix for this subject
    s3_prefix="s3://natural-scenes-dataset/nsddata_betas/ppdata/subj${sub}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    
    # Only loop through the known number of sessions for this subject
    for session in $(seq -f "%02g" 1 ${sessions[$sub]}); do
        echo "Downloading session $session for subject $sub..."
        
        # Construct the full S3 path for this file
        s3_file="$s3_prefix/betas_session${session}.nii.gz"
        
        # Download the file
        if aws s3 cp --no-sign-request "$s3_file" "$target_dir/"; then
            echo "Successfully downloaded session $session for subject $sub"
        else
            echo "Warning: Failed to download session $session for subject $sub"
            # Continue despite errors since we're using set -e
        fi
    done
    
    echo "Completed subject $sub"
done

echo "Finished all subjects in the loop"