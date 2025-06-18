set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/NaturalScenesDataset/GLM"
for sub in {01..08}; do
    python3 ${ROOT}/glmsingle_nsdsynthetic.py -s ${sub} --verbose
done