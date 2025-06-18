set -e
export ROOT=${PROJECT_ROOT}/src/fmriDatasetPreparation/NaturalObjectDataset/temporal_filtering
for subj in {10..30}; do
    echo "Running temporal filtering for sub-${subj}"
    python3 ${ROOT}/nod_temporal_filter.py -s ${subj} -t 'imagenet'
    echo "Finished subject ${subj}"
done
echo "Finished temporal filtering for all subjects in the loop"