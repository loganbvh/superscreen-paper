#!/bin/bash
#SBATCH --job-name=small_squid_psf
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate superscreen

python point_spread_function.py \
    --num_cpus=$SLURM_CPUS_PER_TASK \
    --outdir=$SCRATCH/superscreen-paper/results/ \
    --vortex_height="-0.2" \
    --max_thickness="0.2" \
    --pixel_size="0.25" \
    --job_id $SLURM_JOB_ID
