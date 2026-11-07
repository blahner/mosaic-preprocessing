set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/THINGS_fmri/GLM"
GLM_ROOT="${DATASETS_ROOT}/THINGS_fmri/derivatives/GLM"
for subj in {01..03}; do
echo "Starting glm estimation for sub-${subj}"
    for sesgroup in {1..2}; do
    sg=$(printf "sessiongroup-%02d" ${sesgroup})
    # skip session groups that already finished so the loop can be safely re-run (e.g. after a reboot)
    if [ -f "${GLM_ROOT}/sub-${subj}/${sg}/things/sub-${subj}_${sg}_task-things_conditionOrderDM.pkl" ]; then
        echo "sub-${subj} ${sg} already done - skipping"
        continue
    fi
    echo "Running GLMsingle"
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/glmsingle_things.py -s ${subj} -i ${sesgroup} -v
    done
echo "Finished subject ${subj}"
done
echo "Finished training GLM for all subjects in the loop"
