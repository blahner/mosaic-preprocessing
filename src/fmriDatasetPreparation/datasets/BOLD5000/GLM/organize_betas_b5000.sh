set -e
#source the .env file
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/BOLD5000/GLM"
for sub in {01..04}; do
    python ${ROOT}/b5000_organize_betas_testtrain_combine_sessions.py -s $sub -v
done