function [bin_matrix_movie_sw, bin_matrix_phone_sw, bin_matrix_movie_taps, bin_matrix_phone_taps] = create_binned_data( ...
    top10_filtered_results, taps, movie_start, movie_end, phone_start, phone_end, bin_size_ms, decay_rate, break_threshold_s)

% Bins slow-wave latencies and smartphone tap latencies into fixed-width time bins,
% with optional removal of long breaks in smartphone behavior.
%
% Usage:
%   [bin_matrix_movie_sw, bin_matrix_phone_sw, bin_matrix_movie_taps, bin_matrix_phone_taps] = create_binned_data( ...
%       top10_filtered_results, taps, movie_start, movie_end, phone_start, phone_end, ...
%       bin_size_ms, decay_rate, break_threshold_s);
%
% Inputs:
%   top10_filtered_results : Struct (1x1) with 1x64 struct array 'channels', each having cell 'maxnegpk'.
%   taps                   : Numeric vector of smartphone tap latencies (ms).
%   movie_start, movie_end : Start/end times (ms) for movie condition.
%   phone_start, phone_end : Start/end times (ms) for phone condition.
%   bin_size_ms            : Bin width in ms (must be positive).
%   decay_rate             : Exponential decay multiplier (0 <= decay_rate <= 1).
%   break_threshold_s      : (optional) Threshold in seconds to define behavioral breaks (default = 60s). If <=0, no removal.
%
% Outputs:
%   bin_matrix_movie_sw   : 64 x n_bins matrix of slow-wave data (movie).
%   bin_matrix_phone_sw   : 64 x n_bins matrix of slow-wave data (phone).
%   bin_matrix_movie_taps : 1 x n_bins vector of tap counts (movie).
%   bin_matrix_phone_taps : 1 x n_bins vector of tap counts (phone).

    arguments
        top10_filtered_results (1,1) struct
        taps                  double
        movie_start           (1,1) double
        movie_end             (1,1) double
        phone_start           (1,1) double
        phone_end             (1,1) double
        bin_size_ms           (1,1) double {mustBePositive}
        decay_rate            (1,1) double {mustBeGreaterThanOrEqual(decay_rate,0), mustBeLessThanOrEqual(decay_rate,1)}
        break_threshold_s     (1,1) double = 60
    end

    % Convert threshold to ms and init total gap
    threshold_ms = break_threshold_s * 1000;
    total_gap_ms = 0;

    %---- Optional: excise long breaks in taps and adjust slow-wave times ----
    if break_threshold_s > 0 && numel(taps) > 1
        iti = diff(taps);
        break_idx = find(iti > threshold_ms);
        % Process breaks from last to first to preserve indexing
        for k = numel(break_idx):-1:1
            i = break_idx(k);
            pre_time  = taps(i);
            post_time = taps(i+1);
            gap = post_time - pre_time;
            total_gap_ms = total_gap_ms + gap;

            % Remove slow-wave events in the gap
            for ch = 1:64
                lat_cell = top10_filtered_results.channels(ch).maxnegpk;
                lat_ms   = cell2mat(lat_cell);
                keep     = lat_ms < pre_time | lat_ms >= post_time;
                top10_filtered_results.channels(ch).maxnegpk = num2cell(lat_ms(keep));
            end

            % Merge taps: drop the post-gap tap, keep pre-gap time
            taps(i+1) = [];
            % Shift subsequent taps backward by gap
            taps(i+1:end) = taps(i+1:end) - gap;

            % Shift subsequent slow-wave timestamps backward by gap
            for ch = 1:64
                lat_cell = top10_filtered_results.channels(ch).maxnegpk;
                lat_ms   = cell2mat(lat_cell);
                mask     = lat_ms >= post_time;
                lat_ms(mask) = lat_ms(mask) - gap;
                top10_filtered_results.channels(ch).maxnegpk = num2cell(lat_ms);
            end
        end
        % Adjust phone_end to account for excised breaks
        phone_end = phone_end - total_gap_ms;
    end

    %--------------------------------------------------------------------
    % Internal: bin slow-wave data for a given condition
    %--------------------------------------------------------------------
    function bin_matrix = bin_condition_sw(cond_start, cond_end)
        n_bins = floor((cond_end - cond_start) / bin_size_ms);
        bin_matrix = zeros(64, n_bins);
        for ch = 1:64
            wave_lat_cell = top10_filtered_results.channels(ch).maxnegpk;
            wave_lat_ms   = cell2mat(wave_lat_cell);
            wave_lat_ms   = wave_lat_ms(wave_lat_ms >= cond_start & wave_lat_ms < cond_end);
            last_val = 0;
            for b = 1:n_bins
                b_start = cond_start + (b-1)*bin_size_ms;
                b_end   = b_start + bin_size_ms;
                if any(wave_lat_ms >= b_start & wave_lat_ms < b_end)
                    bin_matrix(ch,b) = 1;
                    last_val = 1;
                else
                    bin_matrix(ch,b) = last_val * decay_rate;
                    last_val = bin_matrix(ch,b);
                end
            end
        end
    end

    %--------------------------------------------------------------------
    % Internal: bin tap data for a given condition
    %--------------------------------------------------------------------
    function tap_bins = bin_condition_taps(cond_start, cond_end)
        n_bins = floor((cond_end - cond_start) / bin_size_ms);
        tap_bins = zeros(1, n_bins);
        for b = 1:n_bins
            b_start = cond_start + (b-1)*bin_size_ms;
            b_end   = b_start + bin_size_ms;
            tap_bins(b) = sum(taps >= b_start & taps < b_end);
        end
    end

    % Generate binned outputs
    bin_matrix_movie_sw   = bin_condition_sw(movie_start, movie_end);
    bin_matrix_phone_sw   = bin_condition_sw(phone_start, phone_end);
    bin_matrix_movie_taps = bin_condition_taps(movie_start, movie_end);
    bin_matrix_phone_taps = bin_condition_taps(phone_start, phone_end);
end