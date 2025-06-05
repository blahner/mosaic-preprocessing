set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/THINGS_fmri/GLM"
for subj in {01..03}; do
echo "Starting glm estimation for sub-${subj}"
    for sesgroup in {1..2}; do
    echo "Running GLMsingle"
    python3 ${ROOT}/glmsingle_things_combined_sessions.py -s ${subj} -i ${sesgroup} -v
    done
echo "Finished subject ${subj}"
done
echo "Finished training GLM for all subjects in the loop"