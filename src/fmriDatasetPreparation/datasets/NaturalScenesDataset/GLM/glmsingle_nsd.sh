set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/NaturalScenesDataset/GLM"
for ses in {01..40}; do
    python3 ${ROOT}/glmsingle_nsd.py -s $1 -i ${ses}
done