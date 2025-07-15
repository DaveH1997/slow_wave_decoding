function export_segments_to_npy(segmented_data, target_path)

% Exports each participant's segmented data to .npy files:
%   - segments.npy  of shape ('total_seg', 'num_channels', 'num_seg_bins')
%   - labels.npy    of shape ('total_seg', 1) indicating condition (0 = movie, 1 = phone)
%   - tap_counts.npy of shape ('total_seg', 1) storing the summed tap counts per segment.
%
% Usage:
%   export_segments_to_npy(segmented_data, 'my_output_folder')
%
% Inputs:
%   segmented_data : Struct with participant IDs as fields. For each participant:
%         .movie_segments    = cell array {'n_movie_seg',1}, each 64 x 'num_seg_bins'
%         .phone_segments    = cell array {'n_phone_seg',1}, each 64 x 'num_seg_bins'
%         .movie_tap_counts  = numeric vector of length 'n_movie_seg'
%         .phone_tap_counts  = numeric vector of length 'n_phone_seg'
%
%   target_path : Directory in which the .npy files are saved. The function will
%                 create participant-specific files named like "XX00_segments.npy",
%                 "XX00_labels.npy", and "XX00_tap_counts.npy".
%
% Outputs:
%   None (files are written to disk).
%
% Requirements:
%   - The function writeNPY (from 'npy-matlab') must be on the MATLAB path.

    arguments
        segmented_data (1,1) struct
        target_path string
    end

    % Ensure the target directory exists
    if ~exist(target_path, 'dir')
        mkdir(target_path);
    end

    % Get list of participant IDs
    participant_ids = fieldnames(segmented_data);

    % Iterate over participants
    for p = 1:numel(participant_ids)
        participant_id = participant_ids{p};

        % Retrieve the cell arrays of segments
        movie_segments_cell = segmented_data.(participant_id).movie_segments;
        phone_segments_cell = segmented_data.(participant_id).phone_segments;
        
        % Retrieve the tap count vectors
        movie_tap_counts_vec = segmented_data.(participant_id).movie_tap_counts;
        phone_tap_counts_vec = segmented_data.(participant_id).phone_tap_counts;

        n_movie_seg = numel(movie_segments_cell);
        n_phone_seg = numel(phone_segments_cell);
        total_seg  = n_movie_seg + n_phone_seg;

        if total_seg == 0
            warning('Participant %s has no segments. Skipping...', participant_id);
            continue;
        end

        % Identify 'num_channels' and 'num_seg_bins' from the first non-empty segment
        sample_seg = [];
        if n_movie_seg > 0
            sample_seg = movie_segments_cell{1};
        elseif n_phone_seg > 0
            sample_seg = phone_segments_cell{1};
        end
        [num_channels, num_seg_bins] = size(sample_seg);  % e.g., [64, 100]

        % Preallocate arrays
        segments = zeros(total_seg, num_channels, num_seg_bins);
        labels = zeros(total_seg, 1);       % 0 = movie, 1 = phone
        tap_counts = zeros(total_seg, 1);

        % Fill in movie segments
        idx = 1;
        for i = 1:n_movie_seg
            seg = movie_segments_cell{i};          % 64 x num_seg_bins
            segments(idx, :, :) = seg;             % place into (idx, :, :)
            labels(idx) = 0;                       % movie label
            tap_counts(idx) = movie_tap_counts_vec(i);
            idx = idx + 1;
        end

        % Fill in phone segments
        for i = 1:n_phone_seg
            seg = phone_segments_cell{i};
            segments(idx, :, :) = seg;
            labels(idx) = 1;                       % phone label
            tap_counts(idx) = phone_tap_counts_vec(i);
            idx = idx + 1;
        end

        % Construct output filenames for this participant
        seg_file    = sprintf('%s_segments.npy', participant_id);
        labels_file = sprintf('%s_labels.npy', participant_id);
        tap_file    = sprintf('%s_tap_counts.npy', participant_id);

        seg_path    = fullfile(target_path, seg_file);
        labels_path = fullfile(target_path, labels_file);
        tap_path    = fullfile(target_path, tap_file);

        % Write the .npy files (using 'writeNPY' from 'npy-matlab')
        writeNPY(segments, seg_path);
        writeNPY(labels, labels_path);
        writeNPY(tap_counts, tap_path);

        fprintf('\nSaved %s, %s, and %s for participant %s\n', seg_file, labels_file, tap_file, participant_id);
    end
end