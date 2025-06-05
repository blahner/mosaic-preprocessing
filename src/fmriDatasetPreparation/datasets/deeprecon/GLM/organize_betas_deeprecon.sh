set -e
# Define root
export ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/deeprecon/GLM"
for sub in {01..03}; do
    python3 ${ROOT}/deeprecon_organize_betas_testtrain_combine_sessions.py -s $sub -v
done