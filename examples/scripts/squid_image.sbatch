#!/bin/bash

#SBATCH --job-name=squid_image
#SBATCH --array=0-80
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G

# This SBATCH script creates a slurm job array where each job gets a single CPU
# which is used to simulate a single row of a SQUID susceptometry image.

echo "slurm script:"
echo "$(<$0)"

echo "python script:"
echo "$(<squid_image_row.py)"

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate superscreen

outdir=$HOME/superscreen-paper/examples/data/$SLURM_ARRAY_JOB_ID
mkdir -p $outdir/logs

python squid_image_row.py \
    --outdir=$outdir \
    --squid-height=-0.045 \
    --min-points=6750 \
    --optimesh-steps=20 \
    --x-range="-4, 4" \
    --y-range="-5, 5" \
    --iterations=5

mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir/logs/
