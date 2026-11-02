#!/bin/bash
# Submit from the project root after creating outputs/: sbatch .dcc/run_full.sbatch
# FULL_RUN_DIR may point to an existing run directory to resume saved work.
#SBATCH --job-name=biostat825-full
#SBATCH --account=scavenger-h200
#SBATCH --partition=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm-%j.log
#SBATCH --error=outputs/slurm-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate RL

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/.runtime/cache/huggingface"
export PIP_CACHE_DIR="$PWD/.runtime/cache/pip"
export MPLCONFIGDIR="$PWD/.runtime/cache/matplotlib"

run_directory="${FULL_RUN_DIR:-outputs/full-${SLURM_JOB_ID}}"
mkdir -p "$run_directory"

# One kernel per phase releases models and preserves the required execution order.
for phase in parts1-6 7a 7b; do
    python -u scripts/run_full_experiments.py --phase "$phase" \
        --output-dir "$run_directory" --kernel RL
done
