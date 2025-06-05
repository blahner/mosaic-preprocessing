set -e
export ROOT=/data/vision/oliva/scratch/datasets/CC2017/scripts
for subj in 2 3; do
python ${ROOT}/check_reproducibility.py -s ${subj} -t 'train' -vp
done