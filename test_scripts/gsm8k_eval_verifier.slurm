#!/bin/bash
#SBATCH --job-name=gsm8k_test_noverifier   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:a5000:1            # number of gpus per node
#SBATCH --mem=16G
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all        # send email whenever job does anything
#SBATCH --mail-user=nancyx@princeton.edu
#SBATCH --output=../eval/gsm8k_eval_verifier.out

module purge
conda init bash
source ~/.bashrc
cd /n/fs/nlp-abiramg/entailment_bank
conda activate entbank
python eval/run_scorer.py --task "task_1" --split test --prediction_file ../NLProofS/prover/lightning_logs/gsm8k_test_noverifier/results_test.tsv --output_dir ../ablation1/ablation1_eval/gsm8k_test_noverifier/ --bleurt_checkpoint ../bleurt-large-512/
