set -e
# Define root. source the .env file first
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/NaturalObjectDataset/GLM"
for sub in {01..30}; do
    python3 ${ROOT}/nod_organize_betas_testtrain_combine_sessions.py -s $sub -v
done
