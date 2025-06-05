set -e
#you should have run 'clip_stimuli.py' already to preprocess the stimuli into 2s chunks
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/CC2017/timeseries_estimates" #your path to the python scripts
for sub in {02..03}; do
    python ${ROOT}/ts_trial_estimates.py --subject $sub --task test -v
    python ${ROOT}/ts_trial_estimates.py --subject $sub --task train -v

    python ${ROOT}/prepareTStrials_step01_cifti.py --subject $sub --task test -vz
    python ${ROOT}/prepareTStrials_step01_cifti.py --subject $sub --task train -vz

    python ${ROOT}/prepareTStrials_step02_cifti.py --subject $sub -vz
done