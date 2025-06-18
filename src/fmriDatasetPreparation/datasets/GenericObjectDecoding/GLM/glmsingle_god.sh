set -e
#make sure you source your .env file before sourcing this script to access the necessary environment variables
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/GenericObjectDecoding/GLM"
for sub in {01..05}; do
    echo "Starting glm estimation for sub-${sub}"
    python3 ${ROOT}/glmsingle_god.py -s $sub -v
    echo "Finished subject ${sub}"
done
