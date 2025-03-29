#!/bin/sh

#SBATCH --job-name=M1e5
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
#SBATCH --time=04:00:00

module load conda
module unload conda
module load conda
conda activate kerr_hackathon_env
module load cuda/12.4.1
python /home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/AAK_Kerr_Comparisons/AAK_Kerr_SNR_comparisons.py


