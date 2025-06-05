set -e
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/THINGS_fmri/temporal_filtering"
for subj in {01..03}; do
    echo "Running temporal filtering for sub-${subj}"
    python3 ${ROOT}/things_temporal_filter.py -s ${subj} -t 'things'
    python3 ${ROOT}/things_temporal_filter.py -s ${subj} -t 'rest'
    python3 ${ROOT}/things_temporal_filter.py -s ${subj} -t '6cat'
    python3 ${ROOT}/things_temporal_filter.py -s ${subj} -t 'pRF'
    echo "Finished subject ${subj}"
done
echo "Finished training GLM for all subjects in the loop"