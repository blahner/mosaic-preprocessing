set -e
#source the .env file to access the environment variables
export ROOT=${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/GenericObjectDecoding/temporal_filtering
for subj in {01..05}; do
    echo "Running temporal filtering for sub-${subj}"
    python3 ${ROOT}/god_temporal_filter.py -s ${subj} -t 'imagery'
    python3 ${ROOT}/god_temporal_filter.py -s ${subj} -t 'perception'
    echo "Finished subject ${subj}"
done
echo "Finished temporal filtering for all subjects in the loop"