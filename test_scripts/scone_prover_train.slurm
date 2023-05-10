#!/bin/bash
#SBATCH --job-name=scone_train     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:a5000:1            # number of gpus per node
#SBATCH --mem=16G
#SBATCH --mail-type=all          # email whenever anything happens
#SBATCH --mail-user=nancyx@princeton.edu
#SBATCH --output=scone_prover_train.output

module purge
conda init bash
source ~/.bashrc
cd /n/fs/nlp-abiramg/NLProofS
conda activate nlproofs
cd prover
python main.py fit --config scone_task2.yaml 
