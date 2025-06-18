set -e
# Define root
export ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/deeprecon/GLM"
for sub in {01..03}; do
    python3 ${ROOT}/organize_betas_deeprecon.py -s $sub -v
done