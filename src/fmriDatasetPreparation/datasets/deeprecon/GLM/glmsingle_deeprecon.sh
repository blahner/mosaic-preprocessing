set -e
# Define root. Source the project's .env file first to access proejct root variable.
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/deeprecon/GLM"
GLM_ROOT="${DATASETS_ROOT}/deeprecon/derivatives/GLM"
for sub in {01..03}; do
    for sg in perceptionArtificialImage perceptionLetterImage perceptionNaturalImageTest perceptionNaturalImageTraining; do
        # skip session groups that already finished so the loop can be safely re-run (e.g. after a reboot)
        if [ -f "${GLM_ROOT}/sub-${sub}/sessiongroup-${sg}/sub-${sub}_sessiongroup-${sg}_task-perception_conditionOrderDM.pkl" ]; then
            echo "sub-${sub} ${sg} already done - skipping"
            continue
        fi
        uv run --project "${PROJECT_ROOT}" python ${ROOT}/glmsingle_deeprecon.py -s $sub -i $sg -v
    done
done
