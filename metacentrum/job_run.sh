#!/bin/bash
#PBS -N BERT_CHECKER
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=39gb:mem=100gb:scratch_local=100gb
#PBS -m abe
#PBS -j oe

# This script should be run from the your home directory on a frontend server
# Fill these variables in order for the script to work
PROJECT_NAME="typos-correction"
SERVER_LOCATION="praha1"
USERNAME="eliasma7"
DATADIR="/storage/$SERVER_LOCATION/home/$USERNAME/$PROJECT_NAME"
export TMPDIR=$SCRATCHDIR

########################################################################################################################
echo "Task started at $(date)"


test -n "$SCRATCHDIR" || { echo >&2 "SCRATCHDIR is not set!"; exit 1; }

echo "Copying data to $SCRATCHDIR at $(date)"
cp -r "$DATADIR" "$SCRATCHDIR" || { echo >&2 "Error copying data to scratch"; exit 1; }
echo "Data copied at $(date)"

cd "$SCRATCHDIR/$PROJECT_NAME" || { echo >&2 "Failed to enter scratch directory"; exit 1; }

module load mambaforge

echo "Creating conda environment at $(date)"
mamba env create -p "$SCRATCHDIR/tmp_env" -f metacentrum/env.yaml || { echo >&2 "Failed to create Conda environment"; exit 1; }
source activate "$SCRATCHDIR/tmp_env" || { echo >&2 "Failed to activate Conda environment"; exit 1; }
echo "Environment created at $(date)"

mkdir -p "$SCRATCHDIR/tmp_env/lib/python3.13/site-packages/neuspell_data/checkpoints" || { echo >&2 "Failed to create checkpoints directory"; exit 1; }
cp -r "$DATADIR/checkpoints/subwordbert-probwordnoise" "$SCRATCHDIR/tmp_env/lib/python3.13/site-packages/neuspell_data/checkpoints" || { echo >&2 "Failed to copy checkpoint"; exit 1; }

echo "Starting model execution at $(date)"
python test.py || { echo >&2 "Python script failed"; exit 1; }

cp "$SCRATCHDIR/$PROJECT_NAME/results.txt" "$DATADIR/results_$(date +'%d_%m_%Y').txt"

source_file="$SCRATCHDIR/tmp_env/lib/python3.13/site-packages/neuspell_data/checkpoints/finetuned_model"
if [ -e "$source_file" ]; then
  cp -r "$source_file" "$DATADIR/models_$(date)"
else
  echo "Source file does not exist."
fi




echo "Task finished at $(date)"