#!/bin/bash

#SBATCH --job-name=LS_sw_pars
#SBATCH --output=LS_sw_pars.out
#SBATCH --error=LS_sw_pars.err
#SBATCH --mail-user="s3821013@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-long"
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

module load MATLAB/2023b
cd /home/s3821013
matlab -nosplash -nodesktop -r "addpath('/home/s3821013/eeglab2023.1'); addpath('/home/s3821013/fieldtrip-20240110'); addpath('/home/s3821013/pipeline'); addpath(genpath('/home/s3821013/src')); run_paired_t_test_pipeline; exit"