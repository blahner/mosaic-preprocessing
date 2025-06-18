set -e
export ROOT=${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/HumanActionsDataset/temporal_filtering
for subj in {01..30}; do
    echo "Running temporal filtering for sub-${subj}"
    python3 ${ROOT}/had_temporal_filter.py -s ${subj} -t 'action'
    echo "Finished subject ${subj}"
done
echo "Finished training GLM for all subjects in the loop"