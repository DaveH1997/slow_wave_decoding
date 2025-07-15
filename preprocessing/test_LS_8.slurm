#!/bin/bash

#SBATCH --job-name=test_LS
#SBATCH --output=test_LS.out
#SBATCH --error=test_LS.err
#SBATCH --mail-user="s3821013@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-short"
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
 
module load MATLAB/2023b
cd /home/s3821013
matlab -nosplash -nodesktop -r "addpath(genpath('.')); eeglab nogui; EEG = pop_loadset('filename', '12_57_07_05_18.set', 'filepath', '/data1/s3821013/local_sleep/AT08'); preprocessed_EEG = preprocessing_complete(EEG); results = twalldetectnew_TA_v4(preprocessed_EEG.data, 1000, 0); top10_filtered_results = filter_results(results); save('top10_filtered_results.mat', 'top10_filtered_results'); wave_pars = compute_wave_pars(top10_filtered_results, EEG.srate); visualize_wave_pars(wave_pars, EEG.chanlocs); exit"