set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
LOCAL_DIR="${PRs
OJECT_ROOT}/src/fmriDatasetPreparation/datasets/NaturalScenesDataset/validation"
echo "LOCAL_DIR: ${LOCAL_DIR}"

for sub in {01..08}; do
    mkdir -p $LOCAL_DIR/output/ncsnr_original/sub-${sub}
    aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata_betas/ppdata/subj${sub}/fsaverage/betas_fithrf_GLMdenoise_RR/lh.ncsnr.mgh \
    $LOCAL_DIR/output/ncsnr_original/sub-${sub}
    aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata_betas/ppdata/subj${sub}/fsaverage/betas_fithrf_GLMdenoise_RR/rh.ncsnr.mgh \
    $LOCAL_DIR/output/ncsnr_original/sub-${sub}
done
echo "Finished all subjects in the loop"