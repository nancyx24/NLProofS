#!/bin/bash
#SBATCH --job-name=cli_entailmentbank_task2      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --mem=16G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=abiramg@princeton.edu
#SBATCH --out="task2_verifier".out

module purge
conda init bash
source ~/.bashrc
cd /n/fs/nlp-abiramg/NLProofS
conda activate nlproofs
cd verifier
python main.py fit --config cli_entailmentbank_task2-small.yaml