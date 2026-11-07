set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/BOLDMomentsDataset/GLM" 
GLM_ROOT="${DATASETS_ROOT}/BOLDMomentsDataset/derivatives/versionC/GLM"
for subj in {01..10}; do
echo "Starting glm estimation for sub-${subj}"
    for ses in {02..05}; do
    # skip sessions that already finished so the loop can be safely re-run (e.g. after a reboot)
    if [ -f "${GLM_ROOT}/sub-${subj}/ses-${ses}/sub-${subj}_betas-typed.dscalar.nii" ]; then
        echo "sub-${subj} ses-${ses} already done - skipping"
        continue
    fi
    echo "Running GLMsingle"
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/glmsingle_bmd.py -s ${subj} -i ${ses} -v
    done
    echo "Finished subject ${subj}"
done
echo "Finished training GLM for all subjects in the loop"
