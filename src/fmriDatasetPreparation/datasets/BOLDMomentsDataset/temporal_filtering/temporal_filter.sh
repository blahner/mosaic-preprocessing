set -e
#this shell script uses relative path to run the python script. They should be in the same directory
for subj in {01..10}; do
    echo "Running temporal filtering for sub-${subj}"
    python3 temporal_filter.py -s ${subj} -t 'rest'
    python3 temporal_filter.py -s ${subj} -t 'localizer'
    python3 temporal_filter.py -s ${subj} -t 'test'
    python3 temporal_filter.py -s ${subj} -t 'train'
    echo "Finished subject ${subj}"
done
echo "Finished temporal filtering for all subjects in the loop"