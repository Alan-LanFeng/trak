#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 16
#SBATCH --gres=gpu:1
#SBATCH --time 2:00:00
#SBATCH --output=/home/lfeng/task_logs/output_%j.log   
#SBATCH --partition h100
#SBATCH --account=vita
#SBATCH --mem=360G

srun python -u examples/trak_lds.py
