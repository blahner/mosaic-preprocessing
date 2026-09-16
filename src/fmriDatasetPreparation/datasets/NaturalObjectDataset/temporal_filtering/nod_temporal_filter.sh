set -e
export ROOT=${PROJECT_ROOT}/src/fmriDatasetPreparation/NaturalObjectDataset/temporal_filtering
for subj in {01..09}; do
    echo "Running temporal filtering for sub-${subj}"
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/nod_temporal_filter.py -s ${subj} -t 'coco'
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/nod_temporal_filter.py -s ${subj} -t 'prf'
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/nod_temporal_filter.py -s ${subj} -t 'floc'
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/nod_temporal_filter.py -s ${subj} -t 'imagenet'
    echo "Finished subject ${subj}"
done
echo "Finished temporal filtering for all subjects in the loop"