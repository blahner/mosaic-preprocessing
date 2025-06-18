set -e
# Define root. source the .env file first
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/NaturalObjectDataset/GLM"
for sub in {01..30}; do
    python3 ${ROOT}/organize_betas_nod.py-s $sub -v
done
