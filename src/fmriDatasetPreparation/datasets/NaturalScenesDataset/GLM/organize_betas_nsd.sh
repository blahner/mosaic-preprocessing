set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT=${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/NaturalScenesDataset/GLM
for subj in {01..08}; do
echo "Starting glm estimation for sub-${subj}"
    echo "organizing betas"
    python3 ${ROOT}/nsd_organize_betas_testtrain.py -s ${subj}
    echo "Finished subject ${subj}"
done
echo "Finished training GLM for all subjects in the loop"