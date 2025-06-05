set -e
export DERIVATIVES="${DATASETS_ROOT}/BOLDMomentsDataset/derivatives/versionC"

version=24.0.2

mkdir -p $DERIVATIVES/mriqc

docker run \
    --user $(id -u):$(id -g) \
    -it --rm \
    -v $DERIVATIVES/fmriprep:/data:ro \
    -v $DERIVATIVES/mriqc:/out \
    \
    nipreps/mriqc:$version \
    /data /out participant \
    --no-sub \
    --nprocs 32 \
    --omp-nthreads 16

echo "Finished MRIQC on BMD"