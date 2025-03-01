#!/bin/bash
#PBS -N JAMSPELL_Benchmark
#PBS -l walltime=20:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=100gb:scratch_local=100gb:cluster=adan
#PBS -m abe
#PBS -j oe

# This script should be run from the your home directory on a frontend server
PROJECT_NAME="typos-correction"
SERVER_LOCATION="praha1"
USERNAME="eliasma7"
WANDB_API_KEY="373b0d6b94a055bdb3eeb24d46e37f8457028db6"
DATADIR="/storage/$SERVER_LOCATION/home/$USERNAME/$PROJECT_NAME"
CHECKPOINTS="/storage/$SERVER_LOCATION/home/$USERNAME/checkpoints/subwordbert-probwordnoise"
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

# get swig
export PATH=/storage/praha1/home/eliasma7/extract_dir/usr/bin/:$PATH

echo "Copying data to $SCRATCHDIR at $(date)"
cp -r "$DATADIR" "$SCRATCHDIR" || { echo >&2 "Error copying data to scratch"; exit 1; }
echo "Data copied at $(date)"

cd "$SCRATCHDIR/$PROJECT_NAME" || { echo >&2 "Failed to enter scratch directory"; exit 1; }


module load mambaforge

echo "Creating conda environment at $(date)"
mamba env create -p "$SCRATCHDIR/tmp_env" -f metacentrum/env_jamspell.yaml || { echo >&2 "Failed to create Conda environment"; exit 1; }
source activate "$SCRATCHDIR/tmp_env" || { echo >&2 "Failed to activate Conda environment"; exit 1; }
echo "Environment created at $(date)"

wandb login $WANDB_API_KEY || { echo >&2 "Failed to log into wandb"; exit 1; }
echo "Logged in wandb at $(date)"

echo "Starting model benchmarking at $(date)"
python jamspell_benchmark.py || { echo >&2 "Python script failed"; exit 1; }

cp "$SCRATCHDIR/$PROJECT_NAME/benchmark_results.txt" "$DATADIR/../benchmark_jamspell_results_$(date '+%Y_%m_%d_%H').txt"

echo "Task finished at $(date)"