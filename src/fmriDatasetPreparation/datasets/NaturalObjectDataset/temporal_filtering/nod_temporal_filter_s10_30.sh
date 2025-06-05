set -e
export ROOT=/data/vision/oliva/blahner/projects/SheenBrain/fmriDatasetPreparation/NaturalObjectDataset/temporal_filtering
for subj in 16 18; do
    echo "Running temporal filtering for sub-${subj}"
    python3 ${ROOT}/nod_temporal_filter.py -s ${subj} -t 'imagenet'
    echo "Finished subject ${subj}"
done
echo "Finished training GLM for all subjects in the loop"