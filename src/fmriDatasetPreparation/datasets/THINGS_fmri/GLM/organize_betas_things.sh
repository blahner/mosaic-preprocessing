set -e
#source the .env file
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/THINGS_fmri/GLM"
for sub in {01..03}; do
    python ${ROOT}/organize_betas_things.py -s $sub -v
done
