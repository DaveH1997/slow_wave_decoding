#!/bin/bash

#SBATCH --job-name=SW_timelocked
#SBATCH --output=SW_timelocked.out
#SBATCH --error=SW_timelocked.err
#SBATCH --mail-user="s3821013@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-short"
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

module load MATLAB/2023b
cd /home/s3821013
matlab -nosplash -nodesktop -r "addpath('/home/s3821013/eeglab2023.1'); addpath('/home/s3821013/fieldtrip-20240110'); addpath('/home/s3821013/pipelineT'); run_timelocked_pipeline; exit"