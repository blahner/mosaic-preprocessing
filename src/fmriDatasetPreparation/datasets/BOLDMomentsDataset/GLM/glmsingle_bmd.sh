set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/BOLDMomentsDataset/GLM" 
for subj in {01..10}; do
echo "Starting glm estimation for sub-${subj}"
    for ses in {02..05}; do
    echo "Running GLMsingle"
    python3 ${ROOT}/glmsingle_bmd_combined.py -s ${subj} -i ${ses} -v
    done
    echo "Finished subject ${subj}"
done
echo "Finished training GLM for all subjects in the loop"