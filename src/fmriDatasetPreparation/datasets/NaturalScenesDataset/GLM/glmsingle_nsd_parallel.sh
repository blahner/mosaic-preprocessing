#!/bin/bash
set -e
# Define root
export ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/NaturalScenesDataset/GLM"

# Define the arguments for the script
args_batch1=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
args_batch2=("11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
args_batch3=("21" "22" "23" "24" "25" "26" "27" "28" "29" "30")
args_batch4=("31" "32" "33" "34" "35" "36" "37" "38" "39" "40")

# Define the number of CPU cores you want to use
num_cores=4

# Define the function to run the Python script
run_python_script() {
    python3 ${ROOT}/glmsingle_nsd.py -s 5 -i "$1"
}

# Export the function to make it available to parallel
export -f run_python_script

# Run script in parallel 
echo "starting parallel batch 1"
parallel --jobs $num_cores run_python_script ::: "${args_batch1[@]}"
echo "starting parallel batch 2"
parallel --jobs $num_cores run_python_script ::: "${args_batch2[@]}"
echo "starting parallel batch 3"
parallel --jobs $num_cores run_python_script ::: "${args_batch3[@]}"
echo "starting parallel batch 4"
parallel --jobs $num_cores run_python_script ::: "${args_batch4[@]}"
echo "Done with GLM"
