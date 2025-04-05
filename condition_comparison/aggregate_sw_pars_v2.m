function [aggregated_sw_pars] = aggregate_sw_pars_v2(sw_pars, aggregate_f, chanlocs, options)

arguments
    sw_pars struct;
    aggregate_f function_handle;
    chanlocs struct = struct();
    options.exclude_channels double = [];
    options.visualize logical = 1;
end

num_channels = length(chanlocs);
incl_channels = setdiff(1:num_channels, options.exclude_channels);

% Initialize the output structure
aggregated_sw_pars = struct();

% Initialize arrays to store all participants' data for each parameter
all_wvspermin_movie = [];
all_p2pamp_movie = [];
all_dslope_movie = [];
all_uslope_movie = [];

all_wvspermin_phone = [];
all_p2pamp_phone = [];
all_dslope_phone = [];
all_uslope_phone = [];

all_wvspermin_overall = [];
all_p2pamp_overall = [];
all_dslope_overall = [];
all_uslope_overall = [];

% Get all participant IDs
participant_ids = fieldnames(sw_pars);
num_participants = numel(participant_ids);

% Loop through each participant to collect their data
for i = 1:num_participants
    participant_id = participant_ids{i};
    
    % Collect movie condition parameters
    all_wvspermin_movie = [all_wvspermin_movie; sw_pars.(participant_id).wave_pars_movie.wvspermin];
    all_p2pamp_movie = [all_p2pamp_movie; sw_pars.(participant_id).wave_pars_movie.p2pamp];
    all_dslope_movie = [all_dslope_movie; sw_pars.(participant_id).wave_pars_movie.dslope];
    all_uslope_movie = [all_uslope_movie; sw_pars.(participant_id).wave_pars_movie.uslope];
    
    % Collect phone condition parameters
    all_wvspermin_phone = [all_wvspermin_phone; sw_pars.(participant_id).wave_pars_phone.wvspermin];
    all_p2pamp_phone = [all_p2pamp_phone; sw_pars.(participant_id).wave_pars_phone.p2pamp];
    all_dslope_phone = [all_dslope_phone; sw_pars.(participant_id).wave_pars_phone.dslope];
    all_uslope_phone = [all_uslope_phone; sw_pars.(participant_id).wave_pars_phone.uslope];
    
    % Collect overall condition parameters
    all_wvspermin_overall = [all_wvspermin_overall; sw_pars.(participant_id).wave_pars_overall.wvspermin];
    all_p2pamp_overall = [all_p2pamp_overall; sw_pars.(participant_id).wave_pars_overall.p2pamp];
    all_dslope_overall = [all_dslope_overall; sw_pars.(participant_id).wave_pars_overall.dslope];
    all_uslope_overall = [all_uslope_overall; sw_pars.(participant_id).wave_pars_overall.uslope];
end

% Apply the aggregation function to each parameter across participants
aggregated_sw_pars.wave_pars_movie.wvspermin = aggregate_f(all_wvspermin_movie, 1);
aggregated_sw_pars.wave_pars_movie.p2pamp = aggregate_f(all_p2pamp_movie, 1);
aggregated_sw_pars.wave_pars_movie.dslope = aggregate_f(all_dslope_movie, 1);
aggregated_sw_pars.wave_pars_movie.uslope = aggregate_f(all_uslope_movie, 1);

aggregated_sw_pars.wave_pars_phone.wvspermin = aggregate_f(all_wvspermin_phone, 1);
aggregated_sw_pars.wave_pars_phone.p2pamp = aggregate_f(all_p2pamp_phone, 1);
aggregated_sw_pars.wave_pars_phone.dslope = aggregate_f(all_dslope_phone, 1);
aggregated_sw_pars.wave_pars_phone.uslope = aggregate_f(all_uslope_phone, 1);

aggregated_sw_pars.wave_pars_overall.wvspermin = aggregate_f(all_wvspermin_overall, 1);
aggregated_sw_pars.wave_pars_overall.p2pamp = aggregate_f(all_p2pamp_overall, 1);
aggregated_sw_pars.wave_pars_overall.dslope = aggregate_f(all_dslope_overall, 1);
aggregated_sw_pars.wave_pars_overall.uslope = aggregate_f(all_uslope_overall, 1);

if options.visualize

    % Create topographical plots for the aggregated parameters

    min_density = floor(min([aggregated_sw_pars.wave_pars_movie.wvspermin, aggregated_sw_pars.wave_pars_phone.wvspermin]));
    max_density = ceil(max([aggregated_sw_pars.wave_pars_movie.wvspermin, aggregated_sw_pars.wave_pars_phone.wvspermin]));

    min_p2pamp = floor(min([aggregated_sw_pars.wave_pars_movie.p2pamp, aggregated_sw_pars.wave_pars_phone.p2pamp]));
    max_p2pamp = ceil(max([aggregated_sw_pars.wave_pars_movie.p2pamp, aggregated_sw_pars.wave_pars_phone.p2pamp]));

    min_dslope = floor(min([aggregated_sw_pars.wave_pars_movie.dslope, aggregated_sw_pars.wave_pars_phone.dslope]));
    max_dslope = ceil(max([aggregated_sw_pars.wave_pars_movie.dslope, aggregated_sw_pars.wave_pars_phone.dslope]));

    min_uslope = floor(min([aggregated_sw_pars.wave_pars_movie.uslope, aggregated_sw_pars.wave_pars_phone.uslope]));
    max_uslope = ceil(max([aggregated_sw_pars.wave_pars_movie.uslope, aggregated_sw_pars.wave_pars_phone.uslope]));

    visualize_wave_pars_new_v2(aggregated_sw_pars.wave_pars_movie, chanlocs(incl_channels), 'm', ...
        'clim_density', [min_density, max_density], ...
        'clim_p2pamp', [min_p2pamp, max_p2pamp], ...
        'clim_dslope', [min_dslope, max_dslope], ...
        'clim_uslope', [min_uslope, max_uslope]);

    visualize_wave_pars_new_v2(aggregated_sw_pars.wave_pars_phone, chanlocs(incl_channels), 'p', ...
        'clim_density', [min_density, max_density], ...
        'clim_p2pamp', [min_p2pamp, max_p2pamp], ...
        'clim_dslope', [min_dslope, max_dslope], ...
        'clim_uslope', [min_uslope, max_uslope]);
    
    visualize_wave_pars_new_v2(aggregated_sw_pars.wave_pars_overall, chanlocs(incl_channels), 'o');

end

end