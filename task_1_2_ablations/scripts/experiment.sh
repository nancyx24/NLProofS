#!/bin/bash

for i in {1..5} 
do 
    sbatch --export=curr=i --job-name="${1}_${i}"  --output="../outputs/${1}/${1}_${i}.out" "${1}".slurm 
done