set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/process_nans"

python3 ${ROOT}/fill_nans_fmri_noiseceiling.py --dataset BMD --fillvalue adjacency -v
python3 ${ROOT}/fill_nans_fmri_noiseceiling.py --dataset BOLD5000 --fillvalue adjacency -v
python3 ${ROOT}/fill_nans_fmri_noiseceiling.py --dataset GOD --fillvalue adjacency -v
python3 ${ROOT}/fill_nans_fmri_noiseceiling.py --dataset THINGS --fillvalue adjacency -v
python3 ${ROOT}/fill_nans_fmri_noiseceiling.py --dataset NOD --fillvalue adjacency -v
python3 ${ROOT}/fill_nans_fmri_noiseceiling.py --dataset NSD --fillvalue adjacency -v
python3 ${ROOT}/fill_nans_fmri_noiseceiling.py --dataset deeprecon --fillvalue adjacency -v

echo "Done with filling noiseceiling nans"