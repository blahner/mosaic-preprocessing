set -e
export ROOT=/data/vision/oliva/scratch/datasets/CC2017/scripts
for subj in 3; do
echo "starting subject ${subj}"
python ${ROOT}/prepareTStrials_step01_cifti.py -s ${subj} -t 'test' -vz
python ${ROOT}/prepareTStrials_step01_cifti.py -s ${subj} -t 'train' -vz
python ${ROOT}/prepareTStrials_step02_cifti.py -s ${subj} -vpz
done
echo "Done with all subjects in loop"