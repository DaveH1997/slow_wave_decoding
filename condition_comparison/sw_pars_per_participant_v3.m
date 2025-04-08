function sw_pars = sw_pars_per_participant_v3(data_path, use_summary_file, chanlocs, options)

% Computes slow-wave parameters per participant, performs outlier detection and interpolation if required,
% and optionally visualizes the results.
%
% This function can operate in two modes:
%
% Usage:
%   sw_pars = sw_pars_per_participant_v3(checkpoint_dir, false, chanlocs, options)
%       - Mode 1 (use_summary_file == false): Loads checkpoint files from 
%         'checkpoint_dir', extracts required slow-wave info, builds a summary
%         struct 'top10_SWs', saves it to a target path, and computes the slow-wave
%         parameters.
%
%   sw_pars = sw_pars_per_participant_v3(summary_file_dir, true, chanlocs, options)
%       - Mode 2 (use_summary_file == true): Loads an existing summary file 
%         ('top10_SWs.mat') from 'summary_file_dir' and computes the slow-wave parameters.
%
% Inputs:
%   data_path (char)
%       - Case 1 (use_summary_file == false): Directory containing checkpoint .mat files.
%       - Case 2 (use_summary_file == true) : Path to an existing 'top10_SWs.mat' summary file.
%
%   use_summary_file (logical)
%       Flag indicating whether a summary file already exists.
%
%   chanlocs (struct)
%       Channel locations used for topographical visualization.
%
%   options (struct) with the following fields:
%       target_path          (string, optional)  : Directory where 'top10_SWs.mat' will be saved if created.
%       participant_stop     (double, optional)  : Number of participants to process before stopping.
%       exclude_channels     (double array, optional): Channels to exclude from analysis.
%       interpolate_outliers (logical, optional) : Whether to interpolate outlier channels.
%       visualize            (logical, optional) : Whether to generate and save topographical plots.
%
% Outputs:
%   sw_pars (struct)
%       Struct containing computed slow-wave parameters for each participant.
%
% Requirements:
%   The functions compute_wave_pars_new_v2, estimate_lambda, boxcox_transform, eeg_emptyset,
%   eeg_interp, and visualize_wave_pars_new_v2 must be in MATLAB's path.

arguments
    data_path char;
    use_summary_file (1,1) logical;
    chanlocs struct = struct();
    options.target_path string = string(pwd);
    options.participant_stop double = [];
    options.exclude_channels double = [];
    options.interpolate_outliers logical = 0;
    options.visualize logical = 1;
end

% If visualization is enabled, create directory for visualization outputs
if options.visualize
    if ~exist('Participant_Topoplots', 'dir')
        mkdir('Participant_Topoplots');
    end
end

% Initialize summary struct 'top10_SWs' and output struct 'sw_pars'
top10_SWs = struct();
sw_pars = struct();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODE SELECTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~use_summary_file

    % MODE 1 - Build summary struct 'top10_SWs' from checkpoint files

    % Initialize participant counter for logging stop condition
    participant_counter = 0;

    % Define target duration for each condition (60 minutes in ms)
    target_duration = 3600000;  % 3,600,000 ms

    % Get list of all .mat files in the specified data path
    file_list = dir(fullfile(data_path, '*.mat'));
    total_files = length(file_list);
    
    % Loop through each file using its index for file counting
    for i = 1:total_files

        % Get file name
        file_name = file_list(i).name;
        
        % Print file count and name
        fprintf('\n---------------------------------------------\n');
        fprintf('File Number: %d / %d\n', i, total_files);
        fprintf('File Name: %s\n', file_name);
        fprintf('---------------------------------------------\n');
        
        % Skip 'EEG_res.mat' file
        if strcmp(file_name, 'EEG_res.mat')
            fprintf('Skipping file: %s\n', file_name);
            continue;
        end

        % Load checkpoint file data
        loaded_data = load(fullfile(data_path, file_name));
        if ~isfield(loaded_data, 'res') || ~iscell(loaded_data.res)
            fprintf('\nFile %s does not contain a "res" cell array. Skipping.\n', file_name);
            continue;
        end
        R = loaded_data.res;
        [num_rows, num_cols] = size(R);
        
        % Iterate over each row of R (each representing a participant instance)
        for row = 1:num_rows

            % Ensure that at least 4 columns exist (expected structure: 1x4; extra columns ignored)
            if num_cols < 4
                fprintf('\nRow %d in file %s does not have enough columns. Skipping.\n', row, file_name);
                continue;
            end
            
            % Check if the second column contains a valid EEG struct with 'filepath'
            if ~isstruct(R{row,2}) || ~isfield(R{row,2}, 'filepath')
                fprintf('\nRow %d in file %s does not contain valid participant data. Skipping.\n', row, file_name);
                continue;
            end
            
            % Extract participant ID based on the last 4 characters of the filepath
            pid = R{row,2}.filepath(end-3:end);
            
            % Increment global participant count (for optional stop condition)
            participant_counter = participant_counter + 1;
            
            % Separate movie and phone indices from the participant data
            [movie_indexes, phone_indexes, ~, ~, taps] = seperate_movie_phone(R(row, :));
            
            % Check 'movie_indexes' and skip if empty
            if isempty(movie_indexes)
                fprintf('\nSkipping participant %s due to missing movie indexes.\n', pid);
                continue;
            end

            % Check 'phone_indexes' and skip if empty
            if isempty(phone_indexes)
                fprintf('\nSkipping participant %s due to missing phone indexes.\n', pid);
                continue;
            end

            % Check 'taps{1}' and skip if empty
            if isempty(taps{1})
                fprintf('\nSkipping participant %s due to missing smartphone tap latencies.\n', pid);
                continue;
            end
            
            % Build participant summary object, extract 'top10_filtered_results' and add it
            participant_data.top10_filtered_results = R{row,4};

            % Add smartphone tap latencies to the participant summary object
            participant_data.taps = taps{1};

            % Extract movie start/end times and add them to participant summary object
            if isfield(movie_indexes{1}, 'movie_latencies')
                participant_data.movie_start = movie_indexes{1}.movie_latencies(1);
                participant_data.movie_end = movie_indexes{1}.movie_latencies(end);
            else
                participant_data.movie_start = NaN;
                participant_data.movie_end = NaN;
                fprintf('\nWarning: Missing movie start/end times for participant %s. Storing NaN.\n', pid);
            end

            % Extract phone start/end times and add them to participant summary object
            if iscell(phone_indexes) && ~isempty(phone_indexes{1})
                participant_data.phone_start = phone_indexes{1}{1}(1);
                participant_data.phone_end = phone_indexes{1}{end}(end);
            else
                participant_data.phone_start = NaN;
                participant_data.phone_end = NaN;
                fprintf('\nWarning: Missing phone start/end times for participant %s. Storing NaN.\n', pid);
            end

            % Extract recording end time and add it to participant summary object
            if isfield(R{row,2}, 'times')
                participant_data.recording_end = R{row,2}.times(end);
            else
                participant_data.recording_end = NaN;
                fprintf('\nWarning: Missing "times" field for participant %s. Storing NaN for recording end time.\n', pid);
            end
            
            % Check that none of the movie/phone timing fields are NaN
            if isnan(participant_data.movie_start) || isnan(participant_data.movie_end) || ...
               isnan(participant_data.phone_start) || isnan(participant_data.phone_end)
                fprintf('\nSkipping participant %s occurrence due to NaN timing values.\n', pid);
                continue;
            end

            % Compute condition order and require it to be 'movie_phone'
            if participant_data.movie_start < participant_data.phone_start
                condition_order = 'movie_phone';
            else
                condition_order = 'phone_movie';
            end
            if ~strcmp(condition_order, 'movie_phone')
                fprintf('\nSkipping participant %s occurrence due to condition order not being movie_phone.\n', pid);
                continue;
            end

            % Adjust movie end time if necessary
            if participant_data.movie_end > participant_data.phone_start
                participant_data.movie_end = participant_data.phone_start;
            end

            % Compute movie and phone durations
            movie_length = participant_data.movie_end - participant_data.movie_start;
            phone_length = participant_data.phone_end - participant_data.phone_start;

            % Compute total deviation from target condition duration
            total_deviation = abs(movie_length - target_duration) + abs(phone_length - target_duration);

            % Store the total deviation in the participant summary object (for later comparison)
            participant_data.total_deviation = total_deviation;
            
            % If this participant does not already exists in 'top10_SWs', add their summary object
            if ~isfield(top10_SWs, pid)
                top10_SWs.(pid) = participant_data;
                fprintf('\nStored data for participant %s with total deviation %.1f min.\n', pid, participant_data.total_deviation / 1000 / 60);
            % If this participant already exists in 'top10_SWs', compare total condition duration deviation
            else
                % If the new occurrence has a lower total condition duration deviation, update the record
                if participant_data.total_deviation < top10_SWs.(pid).total_deviation
                    top10_SWs.(pid) = participant_data;
                    fprintf('\nUpdated data for participant %s with total condition duration deviation %.1f min.\n', pid, participant_data.total_deviation / 1000 / 60);
                else
                    fprintf('\nDiscarded data for participant %s with total condition duration deviation %.1f min.\n', pid, participant_data.total_deviation / 1000 / 60);
                end
            end
            
            % Check if the participant stop condition is met
            if ~isempty(options.participant_stop) && participant_counter >= options.participant_stop
                break;
            end
        end
        
        % Exit the loop if the stop condition is met
        if ~isempty(options.participant_stop) && participant_counter >= options.participant_stop
            break;
        end
    end
    
    % Save 'top10_SWs.mat' to 'target_path'
    summary_file_path = fullfile(options.target_path, 'top10_SWs.mat');
    save(summary_file_path, 'top10_SWs', '-v7.3');
    fprintf('\nSaved top10_SWs to %s\n', summary_file_path);

else

    % MODE 2 - Load existing summary file 'top10_SWs' from data_path

    % Construct the full path to the summary file
    summary_file_path = fullfile(data_path, 'top10_SWs.mat');

    % Load the summary file from 'summary_file_path'
    fprintf('\nLoading existing summary file from %s...\n', summary_file_path);
    loaded_data = load(summary_file_path, 'top10_SWs');
    if ~isfield(loaded_data, 'top10_SWs')
        error('File %s does not contain the variable top10_SWs.', data_path);
    end
    top10_SWs = loaded_data.top10_SWs;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMMON PROCESSING: Compute slow-wave parameters for each participant
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract participant IDs
participant_ids = fieldnames(top10_SWs);

% Loop through each participant
for p = 1:numel(participant_ids)

    % Get the participant ID
    pid = participant_ids{p};

    % If visualization is enabled, create subdirectory for participant visualization output
    if options.visualize
        sub_dir_path = fullfile('Participant_Topoplots', pid);
        if ~exist(sub_dir_path, 'dir')
            mkdir(sub_dir_path);
        end
    end

    % Extract participant data from summary file
    participant_data = top10_SWs.(pid);
    top10_filtered_results = participant_data.top10_filtered_results;
    movie_start = participant_data.movie_start;
    movie_end   = participant_data.movie_end;
    phone_start = participant_data.phone_start;
    phone_end   = participant_data.phone_end;
    recording_end = participant_data.recording_end;
    
    % Calculate the lengths of movie and phone conditions
    movie_length = movie_end - movie_start;
    phone_length = phone_end - phone_start;
    
    % Print participant info
    fprintf('\n---------------------------------------------\n');
    fprintf('Participant: %s\n\n', pid);
    fprintf('Movie start: %.1f min\n', movie_start / 1000 / 60);
    fprintf('Movie end: %.1f min\n', movie_end / 1000 / 60);
    fprintf('Phone start: %.1f min\n', phone_start / 1000 / 60);
    fprintf('Phone end: %.1f min\n\n', phone_end / 1000 / 60);
    fprintf('Movie length: %.1f min\n', movie_length / 1000 / 60);
    fprintf('Phone length: %.1f min\n\n', phone_length / 1000 / 60);
    fprintf('Recording end: %.1f min\n', recording_end / 1000 / 60);
    fprintf('---------------------------------------------\n\n');
    
    % Determine included and excluded channels
    fields = fieldnames(top10_filtered_results.channels);
    num_fields = length(fields);
    num_channels = length(top10_filtered_results.channels);
    num_excl_channels = length(options.exclude_channels);
    num_incl_channels = num_channels - num_excl_channels;
    incl_channels = setdiff(1:num_channels, options.exclude_channels);
    
    % Create condition-specific wave_info structures by filtering 
    wave_info_movie = top10_filtered_results;
    wave_info_phone = top10_filtered_results;
    wave_info_overall = top10_filtered_results;
    
    % Loop through each channel to filter slow waves based on condition times
    for ch = 1:num_channels
        % Extract negative peak latencies for the current channel
        maxnegpk = cell2mat(top10_filtered_results.channels(ch).maxnegpk);
        
        % Determine indices to keep for movie, phone, and overall conditions
        idx_keep_movie = maxnegpk >= movie_start & maxnegpk < movie_end;
        idx_keep_phone = maxnegpk >= phone_start & maxnegpk < phone_end;
        idx_keep_overall = (maxnegpk >= movie_start & maxnegpk < movie_end) | (maxnegpk >= phone_start & maxnegpk < phone_end);
        
        % Loop through each field to filter data accordingly
        for field_idx = 1:num_fields
            field = fields{field_idx};
            if strcmp(field, 'datalength')
                wave_info_movie.channels(ch).(field) = movie_length;
                wave_info_phone.channels(ch).(field) = phone_length;
                wave_info_overall.channels(ch).(field) = movie_length + phone_length;
            else
                wave_info_movie.channels(ch).(field) = top10_filtered_results.channels(ch).(field)(idx_keep_movie);
                wave_info_phone.channels(ch).(field) = top10_filtered_results.channels(ch).(field)(idx_keep_phone);
                wave_info_overall.channels(ch).(field) = top10_filtered_results.channels(ch).(field)(idx_keep_overall);
            end
        end
    end
    
    % Compute wave parameters for movie, phone, and overall conditions
    fs_orig = 1000;
    tmp.wave_pars_movie = compute_wave_pars_new_v2(wave_info_movie, fs_orig, 'channels', incl_channels);
    tmp.wave_pars_phone = compute_wave_pars_new_v2(wave_info_phone, fs_orig, 'channels', incl_channels);
    tmp.wave_pars_overall = compute_wave_pars_new_v2(wave_info_overall, fs_orig, 'channels', incl_channels);
    
    % Outlier detection & interpolation
    if options.interpolate_outliers
        tmp_fields = fieldnames(tmp);
        num_tmp_fields = length(tmp_fields);
        pars_fields = fieldnames(tmp.wave_pars_movie);
        num_pars_fields = length(pars_fields);
        
        % Loop through each condition (movie/phone/overall)
        for tmp_idx = 1:num_tmp_fields
            tmp_field = tmp_fields{tmp_idx};
            % Loop through each slow-wave parameter
            for pars_idx = 1:num_pars_fields
                pars_field = pars_fields{pars_idx};
                pars_vals = tmp.(tmp_field).(pars_field);
                
                % Perform Box-Cox transformation to normalize data
                lambda_opt = estimate_lambda(pars_vals);
                transformed_pars_vals = boxcox_transform(pars_vals, lambda_opt);
                
                % Perform z-standardization
                z_transformed_pars_vals = (transformed_pars_vals - mean(transformed_pars_vals)) / std(transformed_pars_vals);
                
                % Compute absolute z-values for outlier detection
                abs_z_transformed_pars_vals = abs(z_transformed_pars_vals);
                
                % Set z-score threshold (99% confidence interval)
                z_threshold = 2.58;
                
                % Identify outliers exceeding the z-score threshold
                outliers = abs_z_transformed_pars_vals > z_threshold;
                
                % If there are outliers, proceed with interpolation
                if any(outliers)
                    fprintf('\n+++ %d Outlier(s) detected in "%s": %s +++\n\n', sum(outliers), tmp_field, pars_field);
                    
                    % Get indices of bad channels
                    bad_channels = find(outliers);
                    
                    % Initialize interpolated values with original parameter values
                    interpolated_values = pars_vals;
                    
                    % Prepare the EEG structure for interpolation
                    EEG = eeg_emptyset;
                    EEG.data = pars_vals';
                    EEG.nbchan = num_incl_channels;
                    EEG.chanlocs = chanlocs(incl_channels);
                    EEG.srate = 1;  % Dummy sample rate for interpolation
                    EEG.pnts = 1;
                    EEG.trials = 1;
                    EEG.times = 0;
                    EEG.xmin = 0;
                    EEG.xmax = 0;
                    
                    % Mark bad channels with NaN
                    EEG.data(bad_channels) = NaN;
                    
                    % Perform spherical spline interpolation
                    EEG_interp = eeg_interp(EEG, bad_channels, 'spherical');
                    
                    % Replace outlier values with interpolated values
                    interpolated_values(bad_channels) = EEG_interp.data(bad_channels);
                    
                    % Update the condition-wise wave parameters with interpolated values
                    tmp.(tmp_field).(pars_field) = interpolated_values;
                end
            end
        end
    end
    
    % Assign computed wave parameters to the output structure 'sw_pars'
    sw_pars.(pid).wave_pars_movie = tmp.wave_pars_movie;
    sw_pars.(pid).wave_pars_phone = tmp.wave_pars_phone;
    sw_pars.(pid).wave_pars_overall = tmp.wave_pars_overall;

    % If visualization is enabled, create parameter-wise topographical plots
    if options.visualize
        visualize_wave_pars_new_v2(tmp.wave_pars_movie, chanlocs(incl_channels), 'm', sub_dir_path);
        visualize_wave_pars_new_v2(tmp.wave_pars_phone, chanlocs(incl_channels), 'p', sub_dir_path);
        visualize_wave_pars_new_v2(tmp.wave_pars_overall, chanlocs(incl_channels), 'o', sub_dir_path);
    end
end

end