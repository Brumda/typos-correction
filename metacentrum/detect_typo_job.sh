#!/bin/bash
#PBS -N DETECT_TYPO
#PBS -l walltime=20:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=100gb:scratch_local=100gb
#PBS -m abe
#PBS -j oe

# This script should be run from the your home directory on a frontend server
PROJECT_NAME="typos-correction"
SERVER_LOCATION="praha1"
USERNAME="eliasma7"
DATADIR="/storage/$SERVER_LOCATION/home/$USERNAME/$PROJECT_NAME"
# testing:
# cp -r "/storage/praha1/home/eliasma7/typos-correction" "$SCRATCHDIR"

# export PS1="../\W \$ "
########################################################################################################################
set -e
# Ensure clean_scratch runs on exit, even on error
cleanup() {
    echo "Running clean_scratch at $(date)"
    clean_scratch
}
trap cleanup EXIT

echo "Task started at $(date)"
export TMPDIR=$SCRATCHDIR

test -n "$SCRATCHDIR" || { echo >&2 "SCRATCHDIR is not set!"; exit 1; }

echo "Copying data to $SCRATCHDIR at $(date)"
cp -r "$DATADIR" "$SCRATCHDIR" || { echo >&2 "Error copying data to scratch"; exit 1; }
echo "Data copied at $(date)"

cd "$SCRATCHDIR/$PROJECT_NAME" || { echo >&2 "Failed to enter scratch directory"; exit 1; }
WANDB_API_KEY=$(cat $DATADIR/../wandb_key)

module load mambaforge

echo "Creating conda environment at $(date)"
mamba env create -p "$SCRATCHDIR/tmp_env" -f metacentrum/detect_typo_env.yaml || { echo >&2 "Failed to create Conda environment"; exit 1; }
source activate "$SCRATCHDIR/tmp_env" || { echo >&2 "Failed to activate Conda environment"; exit 1; }
echo "Environment created at $(date)"

wandb login "$WANDB_API_KEY" || { echo >&2 "Failed to log into wandb"; exit 1; }
echo "Logged in wandb at $(date)"

mkdir pred_typo_models
echo "Starting model execution at $(date)"
python detect_typo_model.py || { echo >&2 "Python script failed"; exit 1; }

cp "$SCRATCHDIR/$PROJECT_NAME/typo_detect_result.txt" "$DATADIR/../typo_detect_result$(date '+%Y_%m_%d_%H').txt"

source_file="$SCRATCHDIR/$PROJECT_NAME/pred_typo_models"
cp -r "$source_file" "$DATADIR/pred_typo_models_$(date '+%Y_%m_%d_%H')" || { echo >&2 "Source file does not exist."; exit 1; }

echo "Task finished at $(date)"