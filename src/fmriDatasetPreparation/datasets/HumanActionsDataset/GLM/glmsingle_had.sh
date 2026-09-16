set -e
ROOT=${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/HumanActionsDataset/GLM
for subj in {01..30}; do
    echo "Starting glm estimation for sub-${subj}"
    echo "Running GLMsingle"
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/glmsingle_had.py -s ${subj} -v
    echo "Finished subject ${subj}"
done
echo "Finished GLM for all subjects in the loop"