set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/BOLD5000/temporal_filtering"
for subj in {1..4}; do
    echo "Running temporal filtering for sub-${subj}"
    python3 ${ROOT}/temporal_filter.py -s ${subj} -t '5000scenes'
    python3 ${ROOT}/temporal_filter.py -s ${subj} -t 'localizer'
    echo "Finished subject ${subj}"
done
echo "Finished training GLM for all subjects in the loop"