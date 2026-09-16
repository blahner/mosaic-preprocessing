set -e
#source the .env variable first
#note that the order of copying the stimulus sets matters - later copies overwrite the earlier copies in the case of different datasets having 
#the same stimulus. Note that this dataset compilation does not distinguish between different crops of the same stimulus.
ROOT="${PROJECT_ROOT}/src/stimulusSetPreparation/compile_datasets"

uv run --project "${PROJECT_ROOT}" python copy_dataset_stim.py --dataset BOLDMomentsDataset
uv run --project "${PROJECT_ROOT}" python copy_dataset_stim.py --dataset HumanActionsDataset
uv run --project "${PROJECT_ROOT}" python copy_dataset_stim.py --dataset NaturalObjectDataset
uv run --project "${PROJECT_ROOT}" python copy_dataset_stim.py --dataset GenericObjectDecoding
uv run --project "${PROJECT_ROOT}" python copy_dataset_stim.py --dataset deeprecon
uv run --project "${PROJECT_ROOT}" python copy_dataset_stim.py --dataset BOLD5000
uv run --project "${PROJECT_ROOT}" python copy_dataset_stim.py --dataset THINGS_fmri
uv run --project "${PROJECT_ROOT}" python copy_dataset_stim.py --dataset NaturalScenesDataset