set -e
export ROOT=/data/vision/oliva/blahner/projects/SheenBrain/fmriDatasetPreparation/NaturalObjectDataset/temporal_filtering
for subj in {02..09}; do
    echo "Running temporal filtering for sub-${subj}"
    python3 ${ROOT}/nod_temporal_filter.py -s ${subj} -t 'coco'
    python3 ${ROOT}/nod_temporal_filter.py -s ${subj} -t 'prf'
    python3 ${ROOT}/nod_temporal_filter.py -s ${subj} -t 'floc'
    python3 ${ROOT}/nod_temporal_filter.py -s ${subj} -t 'imagenet'
    echo "Finished subject ${subj}"
done
echo "Finished training GLM for all subjects in the loop"