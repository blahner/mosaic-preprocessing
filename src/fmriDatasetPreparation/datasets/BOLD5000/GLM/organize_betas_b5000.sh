set -e
#source the .env file
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/BOLD5000/GLM"
for sub in {01..04}; do
    python ${ROOT}/organize_betas_b5000.py -s $sub -v
done