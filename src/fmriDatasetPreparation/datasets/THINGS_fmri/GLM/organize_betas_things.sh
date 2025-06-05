set -e
#source the .env file
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/THINGS_fmri/GLM"
for sub in {01..03}; do
    python ${ROOT}/things_organize_betas_testtrain_combine_sessions.py -s $sub -v
done
