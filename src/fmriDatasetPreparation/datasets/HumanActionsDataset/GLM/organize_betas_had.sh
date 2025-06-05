set -e
export ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/HumanActionsDataset/GLM"
for subj in {01..30}; do
echo "starting subject ${subj}"
python3 ${ROOT}/had_organize_betas_testtrain.py -s ${subj}
echo "Finished subject ${subj}"
done
echo "Finished GLM for all subjects in the loop"