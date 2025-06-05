set -e
export ROOT=/data/vision/oliva/scratch/datasets/CC2017/scripts
for subj in 2 3; do
python ${ROOT}/ts_trial_estimates.py -s ${subj} -t 'test' -v
python ${ROOT}/ts_trial_estimates.py -s ${subj} -t 'train' -v
done