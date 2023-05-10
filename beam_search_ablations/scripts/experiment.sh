#!/bin/bash
for filename in ./*.slurm; do
    sbatch "$filename"
done