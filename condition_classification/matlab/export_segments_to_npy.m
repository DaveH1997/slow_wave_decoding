function export_segments_to_npy(segmented_data, target_path)

% Exports each participant's segmented data to two .npy files:
%   - segments.npy  of shape (num_segments, num_channels, num_seg_bins)
%   - labels.npy    of shape (num_segments, 1)
%
% Usage:
%   export_segments_to_npy(segmented_data, 'my_output_folder')
%
% Inputs:
%   segmented_data (struct)
%       A struct with participant IDs as fields. For each participant:
%         .movie_segments = cell array {n_movie_segments,1}, each 64 x num_seg_bins
%         .phone_segments = cell array {n_phone_segments,1}, each 64 x num_seg_bins
%
%   target_path (char/string)
%       Directory in which the .npy files are saved. The function will create
%       participant-specific files named like "P1_XXXX_segments.npy" and
%       "P1_XXXX_labels.npy".
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

    % Get the list of participants from the struct
    participant_ids = fieldnames(segmented_data);

    for p = 1:numel(participant_ids)
        participant_id = participant_ids{p};

        % Retrieve the cell arrays of movie/phone segments
        movie_segments_cell = segmented_data.(participant_id).movie_segments; 
        phone_segments_cell = segmented_data.(participant_id).phone_segments; 

        n_movie_seg = numel(movie_segments_cell);
        n_phone_seg = numel(phone_segments_cell);
        total_seg  = n_movie_seg + n_phone_seg;

        if total_seg == 0
            warning('Participant %s has no segments. Skipping...', participant_id);
            continue;
        end

        % Identify num_channels and num_seg_bins from the first non-empty segment
        sample_seg = [];
        if n_movie_seg > 0
            sample_seg = movie_segments_cell{1};
        elseif n_phone_seg > 0
            sample_seg = phone_segments_cell{1};
        end

        [num_channels, num_seg_bins] = size(sample_seg);  % e.g., [64, 100]

        % Preallocate a 3D array of shape (total_seg, num_channels, num_seg_bins)
        segments = zeros(total_seg, num_channels, num_seg_bins);

        % Preallocate a 1D label vector (0 = movie, 1 = phone)
        labels = zeros(total_seg, 1);

        % Fill in movie segments
        idx = 1;
        for i = 1:n_movie_seg
            seg = movie_segments_cell{i};          % 64 x num_seg_bins
            segments(idx, :, :) = seg;             % place into (idx, :, :)
            labels(idx) = 0;                       % movie label
            idx = idx + 1;
        end

        % Fill in phone segments
        for i = 1:n_phone_seg
            seg = phone_segments_cell{i};
            segments(idx, :, :) = seg;
            labels(idx) = 1;                       % phone label
            idx = idx + 1;
        end

        % Construct output filenames for this participant
        seg_file    = sprintf('%s_segments.npy', participant_id);
        labels_file = sprintf('%s_labels.npy',   participant_id);

        seg_path    = fullfile(target_path, seg_file);
        labels_path = fullfile(target_path, labels_file);

        % Write the .npy files (using a library function like writeNPY)
        writeNPY(segments, seg_path);
        writeNPY(labels, labels_path);

        fprintf('\nSaved %s and %s\n', seg_file, labels_file);
    end
end