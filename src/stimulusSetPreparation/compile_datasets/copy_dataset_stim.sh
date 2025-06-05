set -e
#source the .env variable first
#note that the order of copying the stimulus sets matters - later copies overwrite the earlier copies in the case of different datasets having 
#the same stimulus. Note that this dataset compilation does not distinguish between different crops of the same stimulus.
ROOT="${PROJECT_ROOT}/src/stimulusSetPreparation/compile_datasets"

python copy_dataset_stim.py --dataset BOLDMomentsDataset
python copy_dataset_stim.py --dataset HumanActionsDataset
python copy_dataset_stim.py --dataset NaturalObjectDataset
python copy_dataset_stim.py --dataset GenericObjectDecoding
python copy_dataset_stim.py --dataset deeprecon
python copy_dataset_stim.py --dataset BOLD5000
python copy_dataset_stim.py --dataset THINGS_fmri
python copy_dataset_stim.py --dataset NaturalScenesDataset