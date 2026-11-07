set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/BOLDMomentsDataset/temporal_filtering"
for subj in {01..10}; do
    echo "Running temporal filtering for sub-${subj}"
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/temporal_filter.py -s ${subj} -t 'rest'
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/temporal_filter.py -s ${subj} -t 'localizer'
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/temporal_filter.py -s ${subj} -t 'test'
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/temporal_filter.py -s ${subj} -t 'train'
    echo "Finished subject ${subj}"
done
echo "Finished temporal filtering for all subjects in the loop"
