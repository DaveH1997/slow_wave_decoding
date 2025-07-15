#!/bin/bash

#SBATCH --job-name=SW_dens_evol
#SBATCH --output=SW_dens_evol.out
#SBATCH --error=SW_dens_evol.err
#SBATCH --mail-user="s3821013@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-short"
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

module load MATLAB/2023b
cd /home/s3821013
matlab -nosplash -nodesktop -r "addpath('/home/s3821013/pipelineD'); sw_density_evol_driver; exit"