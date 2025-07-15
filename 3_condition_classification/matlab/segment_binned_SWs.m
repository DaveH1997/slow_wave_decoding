function segmented_data = segment_binned_SWs(binned_data, num_seg_bins, overlap)

% Segments each participant's binned data (movie, phone) into overlapping chunks,
% and computes the summed tap counts for each segment.
%
% Usage:
%   segmented_data = segment_binned_SWs(binned_data, num_seg_bins, overlap);
%
% Inputs:
%   binned_data : Struct with participant IDs as fields. For each participant,
%                 the following fields must be present:
%                   .bin_matrix_movie_sw  (64 x n_movie_bins)
%                   .bin_matrix_phone_sw  (64 x n_phone_bins)
%                   .bin_matrix_movie_taps (1 x n_movie_bins)
%                   .bin_matrix_phone_taps (1 x n_phone_bins)
%   num_seg_bins: Number of columns (bins) in each segment (e.g., 100).
%   overlap     : Fraction of overlap between consecutive segments, 0 <= overlap < 1.
%
% Outputs:
%   segmented_data : Struct with participant IDs as fields, each containing:
%         .movie_segments    : Cell array of size {n_segments, 1} of 64 x 'num_seg_bins' EEG segments (movie condition)
%         .phone_segments    : Same as 'movie_segments' but for the phone condition.
%         .movie_tap_counts  : Numeric vector of length n_segments; each element is the sum
%                              of taps in the corresponding movie segment.
%         .phone_tap_counts  : Numeric vector of length n_segments; each element is the sum
%                              of taps in the corresponding phone segment.

    arguments
        binned_data (1,1) struct
        num_seg_bins (1,1) double {mustBePositive}
        overlap (1,1) double {mustBeGreaterThanOrEqual(overlap,0), mustBeLessThan(overlap,1)}
    end

    % Initialize the output struct
    segmented_data = struct();

    % Get the list of participant IDs in 'binned_data'
    participant_ids = fieldnames(binned_data);

    % For each participant, segment the movie and phone matrices
    for p = 1:numel(participant_ids)
        participant_id = participant_ids{p};

        % Extract the EEG (slow-wave) matrices for movie and phone conditions
        bin_matrix_movie_sw = binned_data.(participant_id).bin_matrix_movie_sw;
        bin_matrix_phone_sw = binned_data.(participant_id).bin_matrix_phone_sw;
        
        % Extract the tap count vectors for movie and phone conditions
        bin_matrix_movie_taps = binned_data.(participant_id).bin_matrix_movie_taps;
        bin_matrix_phone_taps = binned_data.(participant_id).bin_matrix_phone_taps;

        % Segment the EEG data using the existing helper
        movie_segments = create_segments(bin_matrix_movie_sw, num_seg_bins, overlap);
        phone_segments = create_segments(bin_matrix_phone_sw, num_seg_bins, overlap);
        
        % Segment the tap data by summing over each corresponding segment
        movie_tap_counts = create_segments_tap(bin_matrix_movie_taps, num_seg_bins, overlap);
        phone_tap_counts = create_segments_tap(bin_matrix_phone_taps, num_seg_bins, overlap);

        % Store the resulting cell arrays and numeric vectors in 'segmented_data'
        segmented_data.(participant_id).movie_segments = movie_segments;
        segmented_data.(participant_id).phone_segments = phone_segments;
        segmented_data.(participant_id).movie_tap_counts = movie_tap_counts;
        segmented_data.(participant_id).phone_tap_counts = phone_tap_counts;
    end
end

%----------------------------------------------------------------------
% INTERNAL FUNCTION: create_segments
%----------------------------------------------------------------------
function segments_cell = create_segments(bin_matrix, num_seg_bins, overlap)
    % Creates overlapping segments (64 x 'num_seg_bins') from the given 'bin_matrix' (64 x 'num_total_bins').
    % Overlap is specified as a fraction in [0, 1). A 'step_size' of 'num_seg_bins'*(1-'overlap') is used to move between consecutive segments.

    [num_channels, num_total_bins] = size(bin_matrix);
    if num_channels ~= 64
        warning('Expected 64 rows (channels), but got %d. Proceeding anyway...', num_channels);
    end

    % Determine the step size
    step_size = round(num_seg_bins * (1 - overlap));
    if step_size < 1
        error('Overlap is too high; resulting step_size is < 1.');
    end

    segments_cell = {};
    seg_idx = 0;
    start_col = 1;
    while true
        end_col = start_col + num_seg_bins - 1;
        if end_col > num_total_bins
            break;  % Discard partially occupied segments
        end

        % Extract the segment
        seg_idx = seg_idx + 1;
        segments_cell{seg_idx, 1} = bin_matrix(:, start_col:end_col);

        % Move to the next segment start
        start_col = start_col + step_size;
    end
end

%----------------------------------------------------------------------
% INTERNAL FUNCTION: create_segments_tap
%----------------------------------------------------------------------
function tap_sums = create_segments_tap(tap_vector, num_seg_bins, overlap)
    % Creates overlapping segments from a 1 x 'num_total_bins' 'tap_vector', and returns a numeric vector
    % where each element is the sum of taps within a segment.
    %
    % Inputs:
    %   tap_vector  : 1 x 'num_total_bins' vector of tap counts.
    %   num_seg_bins: Number of bins per segment.
    %   overlap     : Fraction of overlap between segments.
    %
    % Output:
    %   tap_sums    : Numeric vector with the sum of taps in each segment.
    
    [rows, num_total_bins] = size(tap_vector);
    if rows > 1
        error('Expected tap_vector to be a 1 x n_total_bins vector.');
    end

    % Determine the step size
    step_size = round(num_seg_bins * (1 - overlap));
    if step_size < 1
        error('Overlap is too high; resulting step_size is < 1.');
    end

    tap_sums = [];
    start_idx = 1;
    while true
        end_idx = start_idx + num_seg_bins - 1;
        if end_idx > num_total_bins
            break;  % Discard partially occupied segments
        end
        
        % Sum the taps in the current segment
        seg_sum = sum(tap_vector(start_idx:end_idx));
        tap_sums = [tap_sums, seg_sum];  % Append to the output vector
        
        % Move to the next segment start index
        start_idx = start_idx + step_size;
    end
end