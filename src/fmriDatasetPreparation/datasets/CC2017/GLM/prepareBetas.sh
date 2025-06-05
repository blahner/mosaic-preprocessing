set -e
export ROOT=/data/vision/oliva/scratch/datasets/CC2017/scripts
for subj in 1; do
echo "starting subject ${subj}"
#python ${ROOT}/prepareBetas_step01_cifti.py -s ${subj} -b 'typed' -t 'test' -vz
#python ${ROOT}/prepareBetas_step01_cifti.py -s ${subj} -b 'typed' -t 'train' -vz
python ${ROOT}/prepareBetas_step02_cifti.py -s ${subj} -vpz -b 'typed'
done
echo "Done with all subjects in loop"