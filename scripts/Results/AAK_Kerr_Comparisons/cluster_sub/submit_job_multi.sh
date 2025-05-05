#!/bin/sh

#SBATCH --array=1-4
#SBATCH --job-name=SNRs_job_%a
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
#SBATCH --time=08:00:00

ID_TASK=$SLURM_ARRAY_TASK_ID
let "i=ID_TASK-1"

echo "JOB_ARRAY:"${a}

module load conda
conda activate few_paper
module load cuda/12.4.1
python /home/ad/burkeol/work/KerrEccentricEquatorialFigures/scripts/Results/AAK_Kerr_Comparisons/AAK_Kerr_SNR_comparisons.py 0


