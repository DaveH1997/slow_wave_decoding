% Driver to run the data preparation pipeline (for subsequent decoding analyses)

% Define fixed paths
summary_file_dir = '/data1/s3821013';
target_path = '/data1/s3821013';

% Make a new directory for the prepared data
outdir = fullfile(target_path,'prepared_for_decoding');
if ~exist(outdir,'dir'), mkdir(outdir); end

% Bin the slow-wave data
binned_SWs = compile_binned_SWs(summary_file_dir, ...   % behavioral break removal on
    'bin_size_ms', 100, ...
    'decay_rate', 0.95, ...
    'break_threshold_s', 60);

% binned_SWs = compile_binned_SWs(summary_file_dir, ...   % behavioral break removal off
%     'bin_size_ms', 100, ...
%     'decay_rate', 0.95, ...
%     'break_threshold_s', -999);

% Segment the binned slow-wave data
segmented_data_100_bins = segment_binned_SWs(binned_SWs, 100, 0.5);   % for condition classification
segmented_data_25_bins = segment_binned_SWs(binned_SWs, 25, 0.5);   % for behavior classification

% Export the binned and segmented slow-wave data to .npy files
subdir_100 = fullfile(outdir,'100_bins'); if ~exist(subdir_100,'dir'), mkdir(subdir_100); end
export_segments_to_npy(segmented_data_100_bins, subdir_100);
subdir_25 = fullfile(outdir,'25_bins'); if ~exist(subdir_25,'dir'), mkdir(subdir_25); end
export_segments_to_npy(segmented_data_25_bins, subdir_25);