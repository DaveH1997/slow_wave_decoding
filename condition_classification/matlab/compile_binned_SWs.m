function binned_SWs = compile_binned_SWs(summary_file_dir, options)

% This function loads an existing summary file 'top10_SWs.mat' from the specified directory
% 'summary_file_dir' and processes each participant's data by calling 'create_binned_data'.
%
% Usage:
%   binned_SWs = compile_binned_SWs(summary_file_dir);
%       - Loads 'top10_SWs.mat' from 'summary_file_dir' and returns the resulting 'binned_SWs'.
%
% Inputs:
%   summary_file_dir (char)
%       - Directory where the 'top10_SWs.mat' summary file is located.
%
%   options.bin_size_ms (1,1) double [optional, default = 100]
%       Bin width in ms for 'create_binned_data'.
%
%   options.decay_rate (1,1) double [optional, default = 0.95]
%       Exponential decay multiplier for 'create_binned_data'.
%
% Outputs:
%   binned_SWs (struct)
%       A struct with participant IDs as fields, each containing:
%         .bin_matrix_movie (64 x n_bins)
%         .bin_matrix_phone (64 x n_bins)
%
% Requirements:
%   - The function 'create_binned_data.m' (which requires 'bin_size_ms' and 'decay_rate')
%     must be in MATLAB's path.

    arguments
        summary_file_dir char
        options.bin_size_ms (1,1) double = 100
        options.decay_rate (1,1) double = 0.95
    end

    % Prepare storage struct for binned data
    binned_SWs = struct(); 

    % Construct the full path to the summary file
    summary_file_path = fullfile(summary_file_dir, 'top10_SWs.mat');

    % Load the existing summary file from summary_file_path
    loaded_data = load(summary_file_path, 'top10_SWs');
    if ~isfield(loaded_data, 'top10_SWs')
        error('File %s does not contain the variable top10_SWs.', summary_file_path);
    end
        
    top10_SWs = loaded_data.top10_SWs;
        
    % Retrieve the list of participant IDs
    participant_ids = fieldnames(top10_SWs);
        
    % Loop over participants and call create_binned_data
    for p = 1:numel(participant_ids)
        participant_id = participant_ids{p};
        participant_data = top10_SWs.(participant_id);
            
        % Extract fields needed for create_binned_data
        top10_filtered_results = participant_data.top10_filtered_results;
        movie_start = participant_data.movie_start;
        movie_end   = participant_data.movie_end;
        phone_start = participant_data.phone_start;
        phone_end   = participant_data.phone_end;

        [bin_matrix_movie, bin_matrix_phone] = create_binned_data( ...
            top10_filtered_results, ...
            movie_start, movie_end, ...
            phone_start, phone_end, ...
            options.bin_size_ms, ...
            options.decay_rate);

        binned_SWs.(participant_id).bin_matrix_movie = bin_matrix_movie;
        binned_SWs.(participant_id).bin_matrix_phone = bin_matrix_phone;

        fprintf('\nParticipant %s processed.\n', participant_id);
    end

end