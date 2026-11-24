#!/bin/bash
# Runs notebooks/08_radimagenet_pretraining.ipynb on the university cluster (SLURM).
# Edit the lines marked EDIT, then submit from the repository root:  sbatch scripts/run_radimagenet.sbatch
# If the job reaches the time limit, submit it again: the notebook continues from the last saved epoch.
#SBATCH --job-name=radimagenet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/radimagenet_%j.out
##SBATCH --partition=gpu        # EDIT: GPU partition name on your cluster
##SBATCH --account=my_account   # EDIT: if the cluster needs a project account

set -euo pipefail
mkdir -p logs
source ~/ds_env/bin/activate                    # EDIT: your Python environment (or "module load ...")
export RADIMAGENET_DIR=/scratch/$USER/radimagenet   # EDIT: where RadImageNet was copied
export INIT=${INIT:-imagenet}                   # imagenet (30 epochs) or scratch (90 epochs): sbatch --export=ALL,INIT=scratch ...

nvidia-smi
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    notebooks/08_radimagenet_pretraining.ipynb --output "08_radimagenet_pretraining_${INIT}.ipynb"
