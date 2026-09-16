set -e
#this shell script uses relative path to run the python script. They should be in the same directory
for subj in {01..10}; do
    echo "Running temporal filtering for sub-${subj}"
    uv run --project "${PROJECT_ROOT}" python temporal_filter.py -s ${subj} -t 'rest'
    uv run --project "${PROJECT_ROOT}" python temporal_filter.py -s ${subj} -t 'localizer'
    uv run --project "${PROJECT_ROOT}" python temporal_filter.py -s ${subj} -t 'test'
    uv run --project "${PROJECT_ROOT}" python temporal_filter.py -s ${subj} -t 'train'
    echo "Finished subject ${subj}"
done
echo "Finished temporal filtering for all subjects in the loop"