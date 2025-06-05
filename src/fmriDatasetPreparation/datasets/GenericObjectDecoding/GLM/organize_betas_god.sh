set -e
# Define root
export ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/GenericObjectDecoding/GLM"
for sub in {01..05}; do
    python3 ${ROOT}/god_organize_betas_testtrain_combine_sessions.py -s $sub -v
done