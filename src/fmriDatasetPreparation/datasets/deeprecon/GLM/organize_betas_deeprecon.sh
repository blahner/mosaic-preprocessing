set -e
# Define root
export ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/deeprecon/GLM"
for sub in {01..03}; do
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/organize_betas_deeprecon.py -s $sub -v
done