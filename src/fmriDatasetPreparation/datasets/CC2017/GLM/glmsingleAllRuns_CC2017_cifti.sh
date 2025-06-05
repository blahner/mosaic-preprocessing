set -e
export ROOT=/data/vision/oliva/scratch/datasets/CC2017/scripts
for subj in 1; do
python ${ROOT}/glmsingle_CC2017_multipleRuns_cifti.py -s ${subj} -t 'test' -v
python ${ROOT}/glmsingle_CC2017_multipleRuns_cifti.py -s ${subj} -t 'train' -v
done
echo "Done with all subjects in loop"