data_path = '/data1/s3821013';
loaded_data = load(fullfile(data_path, 'top10_SWs.mat'));
top10_SWs = loaded_data.top10_SWs;
participant_ids_list = fieldnames(top10_SWs);

% Compute and plot SW density evolution for each channel separately
for ch = 1:62
    participant_idx = 0;
    for p = 1:length(participant_ids_list)
        participant_idx = participant_idx + 1;
        participant_id = participant_ids_list{p};
        participant_data = top10_SWs.(participant_id);
        [average_counts_movie, average_counts_phone] = density_binning(participant_id, participant_data, ch);
        data_cell_array_dens{participant_idx, 1} = participant_id;
        data_cell_array_dens{participant_idx, 2} = average_counts_movie;
        data_cell_array_dens{participant_idx, 3} = average_counts_phone;
    end
    density_timeseries_aggregated;
end

% Compute and plot SW density evolution averaged across all channels
ch = 1:62;
for p = 1:length(participant_ids_list)
    participant_idx = participant_idx + 1;
    participant_id = participant_ids_list{p};
    participant_data = top10_SWs.(participant_id);
    [average_counts_movie, average_counts_phone] = density_binning(participant_id, participant_data, ch);
    data_cell_array_dens{participant_idx, 1} = participant_id;
    data_cell_array_dens{participant_idx, 2} = average_counts_movie;
    data_cell_array_dens{participant_idx, 3} = average_counts_phone;
end
density_timeseries_aggregated;