set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT=${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/NaturalScenesDataset/temporal_filtering
for subj in {01..08}; do
    echo "Running temporal filtering for sub-${subj}"
    python3 ${ROOT}/nsd_temporal_filter.py -s ${subj} -t 'nsdcore' -v
    python3 ${ROOT}/nsd_temporal_filter.py -s ${subj} -t 'rest' -v
    python3 ${ROOT}/nsd_temporal_filter.py -s ${subj} -t 'nsdsynthetic' -v
    echo "Finished subject ${subj}"
done
echo "Finished temporal filtering for all subjects in the loop"