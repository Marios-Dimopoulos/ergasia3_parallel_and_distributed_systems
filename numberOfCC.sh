#!/bin/bash
#SBATCH --job-name=matlab_conncomp
#SBATCH --output=matlab_res_%j.out
#SBATCH --error=matlab_res_%j.err
#SBATCH --partition=batch        
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=90GB              
#SBATCH --time=00:39:00

module purge
module load matlab/R2025a

matlab -batch "numberOfCC"