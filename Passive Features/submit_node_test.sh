#!/bin/bash

#PBS -S /bin/bash
#PBS -N "node_test"
#PBS -q cpu
#PBS -l select=1:ncpus=10,walltime=00:05:00
#PBS -k eo

##########################################################################
# Smoke test: run node_test.py inside the 'prova' conda environment
##########################################################################

# Move to the directory from which the job was submitted
cd "$PBS_O_WORKDIR"

# Load modules
module load python3

# Initialize conda for this shell session, then activate the env
# (conda activate does not work directly inside a non-interactive shell
#  without sourcing conda.sh first)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prova

# Sanity check: show which python and node we're on
echo "Running on node: $(hostname)"
echo "Using python:    $(which python3)"
echo "Conda env:       $CONDA_DEFAULT_ENV"
echo "-----------------------------------------"

# Run the Python script
python3 ./node_test.py

# Clean up
conda deactivate

sleep 10s
