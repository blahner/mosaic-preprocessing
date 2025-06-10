set -e
#this shell script uses relative path to run the python script. They should be in the same directory
for subj in {01..10}; do
echo "Starting glm estimation for sub-${subj}"
    echo "organizing betas"
    python3 organize_betas.py -s ${subj}
    echo "Finished subject ${subj}"
done
echo "Finished organizing GLMs for all subjects in the loop"