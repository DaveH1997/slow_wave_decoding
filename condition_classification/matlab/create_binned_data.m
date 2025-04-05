function [bin_matrix_movie, bin_matrix_phone] = create_binned_data( ...
    top10_filtered_results, ...
    movie_start, ...
    movie_end, ...
    phone_start, ...
    phone_end, ...
    bin_size_ms, ...
    decay_rate)

% Bins negative peak latencies (maxnegpk) from the 'top10_filtered_results'
% struct into 64 x n_bins matrices (one for movie, one for phone), applying
% an exponential decay after each detected slow wave.
%
% Usage:
%   [bin_matrix_movie, bin_matrix_phone] = create_binned_data( ...
%       top10_filtered_results, movie_start, movie_end, ...
%       phone_start, phone_end, ...
%       bin_size_ms, decay_rate);
%
% Inputs (required):
%   top10_filtered_results : 1x1 struct containing a 1x64 struct 'channels'
%   movie_start, movie_end : Start/end times (in ms) of the movie condition
%   phone_start, phone_end : Start/end times (in ms) of the phone condition
%   bin_size_ms (double)   : Bin width in ms
%   decay_rate (double)    : Exponential decay multiplier (0 < decay_rate <= 1)
%
% Outputs:
%   bin_matrix_movie : 64 x n_bins matrix for the movie condition
%   bin_matrix_phone : 64 x n_bins matrix for the phone condition

    arguments
        top10_filtered_results (1,1) struct
        movie_start (1,1) double
        movie_end (1,1) double
        phone_start (1,1) double
        phone_end (1,1) double
        bin_size_ms (1,1) double {mustBePositive}
        decay_rate (1,1) double {mustBeGreaterThanOrEqual(decay_rate,0), mustBeLessThanOrEqual(decay_rate,1)}
    end

    %--------------------------------------------------------------------
    % Internal function to bin a single condition
    %--------------------------------------------------------------------
    function bin_matrix = bin_condition(condition_start, condition_end)
        
        % Returns a 64 x n_bins matrix for a single condition.
        
        % Compute number of full bins in [condition_start, condition_end)
        n_bins = floor((condition_end - condition_start) / bin_size_ms);
        
        % Preallocate the output matrix
        bin_matrix = zeros(64, n_bins);
        
        % Loop over the 64 channels
        for channel_idx = 1:64
            % Extract negative peak latencies and convert from cell to numeric
            wave_lat_cell = top10_filtered_results.channels(channel_idx).maxnegpk;
            wave_lat_ms   = cell2mat(wave_lat_cell);
            
            % Keep only wave latencies within this condition's time range
            wave_lat_ms = wave_lat_ms(wave_lat_ms >= condition_start & ...
                                      wave_lat_ms < condition_end);
            
            % Initialize the decay value for this channel
            last_val = 0;
            
            % Loop over each bin
            for b_idx = 1:n_bins
                bin_start = condition_start + (b_idx - 1) * bin_size_ms;
                bin_end   = bin_start + bin_size_ms;
                
                % Check if at least one wave peak is in [bin_start, bin_end)
                if any(wave_lat_ms >= bin_start & wave_lat_ms < bin_end)
                    bin_matrix(channel_idx, b_idx) = 1;  % wave detected
                    last_val = 1;                     % reset decay
                else
                    % Apply exponential decay from the previous bin's value
                    bin_matrix(channel_idx, b_idx) = last_val * decay_rate;
                    last_val = bin_matrix(channel_idx, b_idx);
                end
            end
        end
    end

    %--------------------------------------------------------------------
    % Create binned matrices for movie and phone conditions
    %--------------------------------------------------------------------
    bin_matrix_movie = bin_condition(movie_start, movie_end);
    bin_matrix_phone = bin_condition(phone_start, phone_end);

end