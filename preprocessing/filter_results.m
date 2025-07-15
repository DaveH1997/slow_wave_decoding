function [top10_filtered_results] = filter_results(results)

% Filtering steps adopted from Andrillon et al. (2021)

filtered_results = results; % Copy the original results to preserve the structure

for ch = 1:length(filtered_results.channels) % Iterate through each channel
    wave_indices_to_remove = []; % Initialize an array to keep track of waves to remove
    
    for wi = 1:length(filtered_results.channels(ch).maxpospkamp) % Iterate through each wave
        % Filter 1: Check for positive peak > 75 μV
        if filtered_results.channels(ch).maxpospkamp{wi} > 75
            wave_indices_to_remove = [wave_indices_to_remove, wi]; % Mark for removal
            continue; % Move to the next wave
        end
        
        % Filter 2: Check for large-amplitude events (> 150 μV) in extended time window
        maxamp = max(filtered_results.channels(ch).maxampwn{wi}, abs(filtered_results.channels(ch).minampwn{wi}));
        if maxamp > 150
            wave_indices_to_remove = [wave_indices_to_remove, wi]; % Mark for removal
        end
    end
    
    % Remove marked waves from filtered_results
    fields = fieldnames(filtered_results.channels(ch)); % Get all wave parameter fields
    for field_idx = 2:length(fields) % Start at field index 2 because of field "datalength" (length = 1)
        field = fields{field_idx};
        filtered_results.channels(ch).(field)(wave_indices_to_remove) = []; % Remove entries for each field
    end
end

%%

% % Extracting all peak-to-peak amplitudes for each channel
% p2pamps = arrayfun(@(x) abs([x.maxnegpkamp{:}]) + [x.maxpospkamp{:}], filtered_results.channels, 'UniformOutput', false);
% 
% % Flatten the cell array to a single array containing all p2p amplitudes
% all_p2p_amps = cellfun(@(c) c(:)', p2pamps, 'UniformOutput', false); % Convert each cell to a row vector
% all_p2p_amps = [all_p2p_amps{:}]; % Concatenate all row vectors into a single vector
% 
% % Plotting the histogram of peak-to-peak amplitudes
% figure; % Create a new figure
% histogram(all_p2p_amps, 50); % Plot the histogram with 50 bins (adjust as needed)
% title('Distribution of Peak-to-Peak Amplitudes Across All Channels');
% xlabel('Peak-to-Peak Amplitude');
% ylabel('Frequency');

%%

top10_filtered_results = filtered_results; % Initialize a copy of filtered_results for modification

for ch = 1:length(filtered_results.channels) % Iterate through each channel
    current_channel_p2p_amps = abs([filtered_results.channels(ch).maxnegpkamp{:}]) + [filtered_results.channels(ch).maxpospkamp{:}]; % Current channel's p2p amplitudes
    threshold = prctile(current_channel_p2p_amps, 90); % Calculate the 90th percentile as the threshold for the top 10%
    
    waves_to_keep = find(current_channel_p2p_amps >= threshold); % Find indices of waves that are in the top 10%
    
    % Now filter each parameter in the channel to keep only the top 10% waves
    fields = fieldnames(filtered_results.channels(ch)); % Get all wave parameter fields
    for field_idx = 2:length(fields) % Start at field index 2 because of field "datalength" (length = 1)
        field = fields{field_idx};
        top10_filtered_results.channels(ch).(field) = top10_filtered_results.channels(ch).(field)(waves_to_keep);
    end
end

end