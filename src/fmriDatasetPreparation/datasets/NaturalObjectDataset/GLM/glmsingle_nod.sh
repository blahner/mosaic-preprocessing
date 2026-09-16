set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/NaturalObjectDataset/GLM"
for sub in {01..30}; do
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/glmsingle_nod.py -s $sub -v
done