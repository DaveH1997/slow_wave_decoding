% Driver to run the paired t-test pipeline

% Load toolboxes
eeglab nogui;
ft_defaults;

% Load channel locations file
load('chanlocs.mat');

% Define fixed paths
data_path = '/home/s3821013/data_pi-ghosha2/sws_2025_1_4';  % for checkpoint files
% data_path = '/data1/s3821013';                              % for summary file
target_path = '/data1/s3821013';

% Define ocular channels
ocular_channels = [63, 64];

% Use summary file ('top10_SWs.mat'): y/n
use_summary_file = false;

% Interpolate outliers: y/n
interpolate_outliers = false;

% Call the main data processing function
sw_pars = sw_pars_per_participant_v3( ...
    data_path, use_summary_file, chanlocs, ...
    'target_path', target_path, ...
    'exclude_channels', ocular_channels, ...
    'interpolate_outliers', interpolate_outliers);
save('/data1/s3821013/sw_pars.mat', 'sw_pars', '-v7.3');

% Call the pooling function
aggregated_sw_pars = aggregate_sw_pars_v2(sw_pars, @mean, chanlocs, 'exclude_channels', ocular_channels);
save('/data1/s3821013/aggregated_sw_pars.mat', 'aggregated_sw_pars', '-v7.3');

% Call the analysis function
[t_vals, sign_ch] = condition_contrast_v2(sw_pars, chanlocs, 'exclude_channels', ocular_channels);
t_test_results.t_vals = t_vals; t_test_results.sign_ch = sign_ch;
save('/data1/s3821013/t_test_results.mat', 't_test_results', '-v7.3');