set -e
# Define root. Source the project's .env file first to access proejct root variable.
ROOT="${PROJECT_ROOT}/src/fmriDatasetPreparation/datasets/deeprecon/GLM"
for sub in {02..03}; do
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/glmsingle_deeprecon_combine_sessions.py -s $sub -i perceptionArtificialImage -v
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/glmsingle_deeprecon_combine_sessions.py -s $sub -i perceptionLetterImage -v
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/glmsingle_deeprecon_combine_sessions.py -s $sub -i perceptionNaturalImageTest -v
    uv run --project "${PROJECT_ROOT}" python ${ROOT}/glmsingle_deeprecon_combine_sessions.py -s $sub -i perceptionNaturalImageTraining -v
done
