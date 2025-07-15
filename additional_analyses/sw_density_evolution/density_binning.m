function [average_counts_movie, average_counts_phone] = density_binning(participant_id, participant_data, channels)
% Computes binned SW densities for a participant, binning separately for movie and phone conditions.

    % Extract relevant data
    top10_filtered_results = participant_data.top10_filtered_results;
    movie_start = participant_data.movie_start;
    movie_end = participant_data.movie_end;
    phone_start = participant_data.phone_start;
    phone_end = participant_data.phone_end;
    recording_end = participant_data.recording_end;

    % Adjust movie and phone end times if they overlap
    if movie_start < phone_start
        if movie_end > phone_start
            movie_end = phone_start;
        end
    elseif phone_start < movie_start
        if phone_end > movie_start
            phone_end = movie_start;
        end
    end

    % Convert times to seconds and minutes
    movie_start_sec = movie_start / 1000;
    movie_end_sec = movie_end / 1000;
    phone_start_sec = phone_start / 1000;
    phone_end_sec = phone_end / 1000;

    movie_length_sec = movie_end_sec - movie_start_sec;
    phone_length_sec = phone_end_sec - phone_start_sec;

    % Adjust channels data
    num_channels = length(top10_filtered_results.channels);
    for ch = 1:num_channels
        negzx = cell2mat(top10_filtered_results.channels(ch).negzx);
        idx_keep_movie_phone = (negzx >= movie_start & negzx < movie_end) | (negzx >= phone_start & negzx < phone_end);

        channel_fields = fieldnames(top10_filtered_results.channels(ch));
        for field_idx = 1:length(channel_fields)
            field = channel_fields{field_idx};
            if strcmp(field, 'datalength')
                top10_filtered_results.channels(ch).(field) = (movie_length_sec + phone_length_sec) * 1000; % Convert back to ms
            else
                field_data = top10_filtered_results.channels(ch).(field);
                top10_filtered_results.channels(ch).(field) = field_data(idx_keep_movie_phone);
            end
        end
    end

    %% PRINT PARTICIPANT INFO %%

    fprintf('\n---------------------------------------------\n');
    fprintf('Participant: %s\n\n', participant_id);
    fprintf('Movie start: %.1f min\n', movie_start_sec / 60);
    fprintf('Movie end: %.1f min\n', movie_end_sec / 60);
    fprintf('Phone start: %.1f min\n', phone_start_sec / 60);
    fprintf('Phone end: %.1f min\n\n', phone_end_sec / 60);
    fprintf('Movie length: %.1f min\n', movie_length_sec / 60);
    fprintf('Phone length: %.1f min\n\n', phone_length_sec / 60);
    fprintf('Recording end: %.1f min\n', recording_end / 1000 / 60);
    fprintf('---------------------------------------------\n\n');

    %% Binning for Movie Condition %%

    % Calculate number of full bins in movie condition
    num_full_bins_movie = floor(movie_length_sec / 60);

    if num_full_bins_movie >= 1
        % Define bin edges for movie condition
        bin_edges_movie_sec = movie_start_sec : 60 : (movie_start_sec + num_full_bins_movie * 60);
        num_bins_movie = length(bin_edges_movie_sec) - 1;
    else
        warning('Movie duration is less than one full bin for participant %s.', participant_id);
        average_counts_movie = [];
        num_bins_movie = 0;
    end

    % Initialize counts for movie condition
    channel_counts_movie = zeros(length(channels), num_bins_movie);

    % Binning for movie condition
    for ch_idx = 1:length(channels)
        ch = channels(ch_idx);
        maxnegpk_data = top10_filtered_results.channels(ch).maxnegpk;
        maxnegpk_seconds = cell2mat(maxnegpk_data) / 1000;

        % Keep only data within movie condition
        maxnegpk_movie = maxnegpk_seconds(maxnegpk_seconds >= movie_start_sec & maxnegpk_seconds < movie_end_sec);

        % Count the occurrences in each bin for the current channel
        counts = histcounts(maxnegpk_movie, bin_edges_movie_sec);

        % Store counts for each channel
        channel_counts_movie(ch_idx, :) = counts;
    end

    % Average counts across channels for movie condition
    average_counts_movie = mean(channel_counts_movie, 1);

    %% Binning for Phone Condition %%

    % Calculate number of full bins in phone condition
    num_full_bins_phone = floor(phone_length_sec / 60);

    if num_full_bins_phone >= 1
        % Define bin edges for phone condition
        bin_edges_phone_sec = phone_start_sec : 60 : (phone_start_sec + num_full_bins_phone * 60);
        num_bins_phone = length(bin_edges_phone_sec) - 1;
    else
        warning('Phone duration is less than one full bin for participant %s.', participant_id);
        average_counts_phone = [];
        num_bins_phone = 0;
    end

    % Initialize counts for phone condition
    channel_counts_phone = zeros(length(channels), num_bins_phone);

    % Binning for phone condition
    for ch_idx = 1:length(channels)
        ch = channels(ch_idx);
        maxnegpk_data = top10_filtered_results.channels(ch).maxnegpk;
        maxnegpk_seconds = cell2mat(maxnegpk_data) / 1000;

        % Keep only data within phone condition
        maxnegpk_phone = maxnegpk_seconds(maxnegpk_seconds >= phone_start_sec & maxnegpk_seconds < phone_end_sec);

        % Count the occurrences in each bin for the current channel
        counts = histcounts(maxnegpk_phone, bin_edges_phone_sec);

        % Store counts for each channel
        channel_counts_phone(ch_idx, :) = counts;
    end

    % Average counts across channels for phone condition
    average_counts_phone = mean(channel_counts_phone, 1);
end