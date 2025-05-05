#!/bin/sh

#SBATCH --job-name=mismatch_plot
#SBATCH --output=output_logs/output_%j.out
#SBATCH --error=error_logs/error_%j.err
#SBATCH --account=lisa
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_all
#SBATCH --mem=100G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=05:00:00

module load conda
module unload conda
module load conda
conda activate kerr_hackathon_env
module load cuda/12.4.1
python /home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/mismatches_eps_e0/eps_e0_heatmap_mismatch_study.py


