#!/bin/bash
#PBS -N T5_Benchmark
#PBS -l walltime=20:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=20gb:scratch_local=100gb:cluster=adan
#PBS -m abe
#PBS -j oe

# This script should be run from the your home directory on a frontend server
PROJECT_NAME="typos-correction"
SERVER_LOCATION="praha1"
USERNAME="eliasma7"
DATADIR="/storage/$SERVER_LOCATION/home/$USERNAME/$PROJECT_NAME"
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
mamba env create -p "$SCRATCHDIR/tmp_env" -f metacentrum/T5_env.yaml || { echo >&2 "Failed to create Conda environment"; exit 1; }
source activate "$SCRATCHDIR/tmp_env" || { echo >&2 "Failed to activate Conda environment"; exit 1; }
echo "Environment created at $(date)"

wandb login "$WANDB_API_KEY" || { echo >&2 "Failed to log into wandb"; exit 1; }
echo "Logged in wandb at $(date)"


echo "Starting model benchmarking at $(date)"
python T5_Benchmark.py || { echo >&2 "Python script failed"; exit 1; }

cp "$SCRATCHDIR/$PROJECT_NAME/benchmark_results.txt" "$DATADIR/../benchmark_results/t5_$(date '+%Y_%m_%d_%H').txt"  || { echo >&2 "Failed to results"; exit 1; }

echo "Task finished at $(date)"