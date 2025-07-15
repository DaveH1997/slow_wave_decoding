#!/bin/bash

#SBATCH --job-name=SW_data_prep
#SBATCH --output=SW_data_prep.out
#SBATCH --error=SW_data_prep.err
#SBATCH --mail-user="s3821013@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-short"
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

module load MATLAB/2023b
cd /home/s3821013
matlab -nosplash -nodesktop -r "addpath('/home/s3821013/npy-matlab'); addpath('/home/s3821013/pipelineP'); run_data_prep_pipeline; exit"