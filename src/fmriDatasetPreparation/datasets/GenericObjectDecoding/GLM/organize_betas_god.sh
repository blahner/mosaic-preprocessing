set -e
# Define root
export ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/GenericObjectDecoding/GLM"
for sub in {01..05}; do
    python3 ${ROOT}/organize_betas_god.py -s $sub -v
done