
#!/bin/bash
set -e

#declare -A dataset_subjects=(
#    ["BMD"]=10
#    ["HAD"]=30   
#    ["NSD"]=8 
#    ["BOLD5000"]=4 
#    ["NOD"]=30
#    ["deeprecon"]=3 
#    ["THINGS"]=3
#    ["GOD"]=5
#)

declare -A dataset_subjects=(
    ["deeprecon"]=3 
    ["THINGS"]=3
    ["GOD"]=5
)

echo "Starting batch processing..."

# Loop through each dataset and its subjects
for dataset in "${!dataset_subjects[@]}"; do
    subject_count=${dataset_subjects[$dataset]}
    echo "Processing dataset: $dataset (${subject_count} subjects)"
    
    for ((i=1; i<=subject_count; i++)); do
        # Format subject ID with zero padding (adjust padding as needed)
        subjectID_dataset=$(printf "sub-%02d_%s" $i $dataset)
        
        echo "Processing: ${subjectID_dataset}"
        
        # Create HDF5 file
        python create_hdf5.py \
            --subjectID_dataset "${subjectID_dataset}" \
            --owner_name "Benjamin Lahner" \
            --owner_email "blahner@mit.edu"
        
        # Upload to S3
        python upload_hdf5.py \
            --file_path "/data/vision/oliva/datasets/MOSAIC/hdf5_files/${subjectID_dataset}.hdf5"
        
        # Clean up local file
        rm "/data/vision/oliva/datasets/MOSAIC/hdf5_files/${subjectID_dataset}.hdf5"
        
        echo "✅ Finished ${subjectID_dataset}"
    done
    
    echo "Completed dataset: $dataset"
    echo "---"
done

echo "🎉 All datasets processed successfully!"