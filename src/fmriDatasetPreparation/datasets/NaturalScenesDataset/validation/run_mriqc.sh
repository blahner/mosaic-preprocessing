set -e
export ROOT=${DATASETS_ROOT}/NaturalScenesDataset

mkdir -p $ROOT/derivatives/mriqc

docker run \
    --user $(id -u):$(id -g) \
    -it --rm \
    -v $ROOT/derivatives/fmriprep:/data:ro \
    -v $ROOT/derivatives/mriqc:/out \
    \
    nipreps/mriqc:latest \
    /data /out participant \
    --no-sub

echo "Finished MRIQC on NSD"