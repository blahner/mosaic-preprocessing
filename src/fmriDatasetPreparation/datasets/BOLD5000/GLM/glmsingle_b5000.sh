set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/BOLD5000/GLM"
GLM_ROOT="${DATASETS_ROOT}/BOLD5000/derivatives/GLM"
for subj in {1..4}; do
echo "Starting glm estimation for sub-${subj}"
    # sub-CSI4 only has 9 sessions (2 session groups); sub-CSI1-3 have 15 (3 session groups)
    if [ ${subj} -eq 4 ]; then ngroups=2; else ngroups=3; fi
    for sesgroup in $(seq 1 ${ngroups}); do
    sg=$(printf "sessiongroup-%02d" ${sesgroup})
    # skip session groups that already finished so the loop can be safely re-run (e.g. after a reboot)
    if [ -f "${GLM_ROOT}/sub-CSI${subj}/${sg}/5000scenes/sub-CSI${subj}_${sg}_task-5000scenes_conditionOrderDM.pkl" ]; then
        echo "sub-CSI${subj} ${sg} already done - skipping"
        continue
    fi
    echo "Running GLMsingle"
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/glmsingle_b5000.py -s ${subj} -i ${sesgroup} -v
    done
    echo "Finished subject ${subj}"
done
echo "Finished training GLM for all subjects in the loop"
