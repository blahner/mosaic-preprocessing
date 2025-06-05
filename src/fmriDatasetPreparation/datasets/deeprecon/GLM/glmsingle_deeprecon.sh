set -e
# Define root. Source the project's .env file first to access proejct root variable.
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/deeprecon/GLM"
for sub in {02..03}; do
    python3 ${ROOT}/glmsingle_deeprecon_combine_sessions.py -s $sub -i perceptionArtificialImage -v
    python3 ${ROOT}/glmsingle_deeprecon_combine_sessions.py -s $sub -i perceptionLetterImage -v
    python3 ${ROOT}/glmsingle_deeprecon_combine_sessions.py -s $sub -i perceptionNaturalImageTest -v
    python3 ${ROOT}/glmsingle_deeprecon_combine_sessions.py -s $sub -i perceptionNaturalImageTraining -v
done
