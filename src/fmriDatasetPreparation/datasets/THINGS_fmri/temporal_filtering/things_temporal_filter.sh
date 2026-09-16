set -e
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/THINGS_fmri/temporal_filtering"
for subj in {01..03}; do
    echo "Running temporal filtering for sub-${subj}"
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/things_temporal_filter.py -s ${subj} -t 'things'
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/things_temporal_filter.py -s ${subj} -t 'rest'
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/things_temporal_filter.py -s ${subj} -t '6cat'
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/things_temporal_filter.py -s ${subj} -t 'pRF'
    echo "Finished subject ${subj}"
done
echo "Finished temporal filtering for all subjects in the loop"