#!/bin/bash
#SBATCH --job-name=single_shot_t5_small      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --mem=16G
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=single_shot_t5_small.out
#SBATCH --mail-type=all        # send email whenever job does anything
#SBATCH --mail-user=nancyx@princeton.edu

module purge
conda init bash
source ~/.bashrc
cd /n/fs/nlp-abiramg/NLProofS
conda activate nlproofs
cd prover
python main.py fit --config cli_task1_single_shot_t5-small.yaml
