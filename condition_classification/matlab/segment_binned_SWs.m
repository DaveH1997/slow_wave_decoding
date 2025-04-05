function segmented_data = segment_binned_SWs(binned_data, num_seg_bins, overlap)

% Segments each participant's binned data (movie, phone) into overlapping
% chunks of size 64 x num_seg_bins. The overlap is specified as a fraction
% between 0 and 1. For example, if overlap = 0.5 and num_seg_bins = 100,
% consecutive segments share 50 bin columns.
%
% Usage:
%   segmented_data = segment_binned_SWs(binned_data, 100, 0.5);
%     - Produces overlapping segments of size 64 x 100 with 50% overlap
%       for both the movie and phone data of each participant.
%
% Inputs:
%   binned_data (struct)
%       A struct with participant IDs as fields. For each participant,
%       there should be:
%         .bin_matrix_movie (64 x nMovieBins)
%         .bin_matrix_phone (64 x nPhoneBins)
%   num_seg_bins (double)
%       The number of columns (bins) in each segment (e.g., 100).
%   overlap (double)
%       Fraction of overlap between consecutive segments, 0 <= overlap < 1.
%       For example, overlap = 0.5 indicates 50% overlap.
%
% Outputs:
%   segmented_data (struct)
%       A struct with participant IDs as fields, each containing:
%         .movie_segments : cell array of size {nSegments, 1}, where each cell
%                           is a 64 x num_seg_bins matrix for the movie condition.
%         .phone_segments : same format as movie_segments but for the phone condition.
%
% Requirements:
%   - binned_data must contain .bin_matrix_movie and .bin_matrix_phone for
%     each participant.
%   - The size of each bin_matrix must be at least 64 x num_seg_bins to
%     produce at least one segment.

    arguments
        binned_data (1,1) struct
        num_seg_bins (1,1) double {mustBePositive}
        overlap (1,1) double {mustBeGreaterThanOrEqual(overlap,0), mustBeLessThan(overlap,1)}
    end

    % Initialize the output struct
    segmented_data = struct();

    % Get the list of participant IDs in binned_data
    participant_ids = fieldnames(binned_data);

    % For each participant, segment the movie and phone matrices
    for p = 1:numel(participant_ids)
        participant_id = participant_ids{p};

        % Extract the 64 x nBins matrices
        bin_matrix_movie = binned_data.(participant_id).bin_matrix_movie;
        bin_matrix_phone = binned_data.(participant_id).bin_matrix_phone;

        % Segment for the movie data
        movie_segments = create_segments(bin_matrix_movie, num_seg_bins, overlap);

        % Segment for the phone data
        phone_segments = create_segments(bin_matrix_phone, num_seg_bins, overlap);

        % Store the resulting cell arrays in segmented_data
        segmented_data.(participant_id).movie_segments = movie_segments;
        segmented_data.(participant_id).phone_segments = phone_segments;
    end

end % function segment_binned_SWs

%----------------------------------------------------------------------
% INTERNAL FUNCTION: create_segments
%----------------------------------------------------------------------
function segments_cell = create_segments(bin_matrix, num_seg_bins, overlap)

    % Creates overlapping segments (64 x num_seg_bins) from the given
    % bin_matrix (64 x num_total_bins). Overlap is specified as a fraction
    % in [0, 1). A step_size of num_seg_bins*(1-overlap) is used to move
    % between consecutive segments.

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