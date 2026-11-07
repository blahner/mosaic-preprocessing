set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/BOLDMomentsDataset/GLM"
for subj in {01..10}; do
    echo "organizing betas for sub-${subj}"
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/organize_betas.py -s ${subj}
    echo "Finished subject ${subj}"
done
echo "Finished organizing GLMs for all subjects in the loop"
