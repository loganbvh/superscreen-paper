#!/bin/bash
#SBATCH --job-name=susc_sweep_height
#SBATCH --cpus-per-task=20
#SBATCH --mem=8GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate superscreen

python susceptometry.py $SLURM_CPUS_PER_TASK 10

ray memory --sort-by=OBJECT_SIZE
