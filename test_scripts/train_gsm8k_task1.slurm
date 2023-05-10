#!/bin/bash
#SBATCH --job-name=train_gsm8k     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:a5000:1            # number of gpus per node
#SBATCH --mem=16G
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=hbgao@princeton.edu
#SBATCH --output="train_gsm8k".out

module purge
conda init bash
source ~/.bashrc
cd /n/fs/nlp-abiramg/NLProofS
conda activate nlproofs
cd prover
python main.py fit --config train_gsm_task1.yaml
