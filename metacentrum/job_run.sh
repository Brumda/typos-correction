#!/bin/bash
#PBS -N ELMO_CHECKER
#PBS -l walltime=20:0:0
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=44gb:mem=100gb:scratch_local=100gb
#PBS -m abe
#PBS -j oe

# This script should be run from the your home directory on a frontend server
PROJECT_NAME="typos-correction"
SERVER_LOCATION="praha1"
USERNAME="eliasma7"
WANDB_API_KEY="373b0d6b94a055bdb3eeb24d46e37f8457028db6"
DATADIR="/storage/$SERVER_LOCATION/home/$USERNAME/$PROJECT_NAME"
CHECKPOINTS="/storage/$SERVER_LOCATION/home/$USERNAME/checkpoints/elmoscrnn-probwordnoise"
# testing:
# cp -r "/storage/praha1/home/eliasma7/typos-correction" "$SCRATCHDIR"
# cp -r "/storage/praha1/home/eliasma7/checkpoints/elmoscrnn-probwordnoise" "$SCRATCHDIR/tmp_env/lib/python3.9/site-packages/neuspell_data/checkpoints"
# cp -r "/storage/praha1/home/eliasma7/checkpoints/elmoscrnn-probwordnoise
# wandb login 373b0d6b94a055bdb3eeb24d46e37f8457028db6


# export PS1="../\W \$ "
# conda config --set changeps1 False
########################################################################################################################
echo "Task started at $(date)"
export TMPDIR=$SCRATCHDIR


test -n "$SCRATCHDIR" || { echo >&2 "SCRATCHDIR is not set!"; exit 1; }

echo "Copying data to $SCRATCHDIR at $(date)"
cp -r "$DATADIR" "$SCRATCHDIR" || { echo >&2 "Error copying data to scratch"; exit 1; }
echo "Data copied at $(date)"

cd "$SCRATCHDIR/$PROJECT_NAME" || { echo >&2 "Failed to enter scratch directory"; exit 1; }

module load mambaforge

echo "Creating conda environment at $(date)"
mamba env create -p "$SCRATCHDIR/tmp_env" -f metacentrum/env.yaml || { echo >&2 "Failed to create Conda environment"; exit 1; }
source activate "$SCRATCHDIR/tmp_env" || { echo >&2 "Failed to activate Conda environment"; exit 1; }
python -m spacy download en_core_web_sm
echo "Environment created at $(date)"

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

mkdir -p "$SCRATCHDIR/tmp_env/lib/python$PYTHON_VERSION/site-packages/neuspell_data/checkpoints/elmoscrnn-probwordnoise" || { echo >&2 "Failed to create checkpoints directory"; exit 1; }
cp -r "$CHECKPOINTS" "$SCRATCHDIR/tmp_env/lib/python$PYTHON_VERSION/site-packages/neuspell_data/checkpoints" || { echo >&2 "Failed to copy checkpoint"; exit 1; }

wandb login $WANDB_API_KEY || { echo >&2 "Failed to log into wandb"; exit 1; }

echo "Logged in wandb at $(date)"
echo "Starting model execution at $(date)"
python test.py || { echo >&2 "Python script failed"; exit 1; }

cp "$SCRATCHDIR/$PROJECT_NAME/result2.txt" "$DATADIR/../results_elmo_$(date '+%Y_%m_%d_%H').txt"

source_file="$SCRATCHDIR/tmp_env/lib/python$PYTHON_VERSION/site-packages/neuspell_data/checkpoints/elmoscrnn-probwordnoise/finetuned_model"
if [ -e "$source_file" ]; then
  cp -r "$source_file" "$DATADIR/../models_elmo_$(date '+%Y_%m_%d_%H')"
else
  echo "Source file does not exist."
fi

echo "Task finished at $(date)"