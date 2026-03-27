#!/bin/bash
#SBATCH --job-name=ML4SCI_Eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Load Apptainer on the Rider/Pioneer Cluster
module load apptainer

# Create logs directory if it doesn't exist on the cluster
mkdir -p logs

# Execute the evaluation inside the container
# Replace 'YOUR_WANDB_ID' with the actual ID from your training run
apptainer exec --nv containers/Singularity.sif \
    python3 scripts/eval.py \
    --run_id "YOUR_WANDB_ID" \
    --project "ml4sci_deeplense_final"
