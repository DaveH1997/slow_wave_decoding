function [t_vals, sign_ch] = condition_contrast_v2(sw_pars, chanlocs, options)

arguments
    sw_pars struct;
    chanlocs struct;
    options.exclude_channels double = [];
    options.visualize logical = 1;
end

    % Get all participant IDs from the sw_pars structure
    participant_ids = fieldnames(sw_pars);
    num_participants = length(participant_ids);

    % Define (number of) included channels
    num_channels = length(chanlocs);
    num_excl_channels = length(options.exclude_channels);
    num_incl_channels = num_channels - num_excl_channels;
    incl_channels = setdiff(1:num_channels, options.exclude_channels);

    % Initialize output structures for t-values and p-values
    pars = {'wvspermin', 'p2pamp', 'dslope', 'uslope'};
    t_vals = struct();
    sign_ch = struct();
    
    for par_idx = 1:length(pars)
        par_name = pars{par_idx};

        % Initialize matrices to store data of all participants for movie and phone conditions
        movie_data = zeros(num_participants, num_incl_channels);
        phone_data = zeros(num_participants, num_incl_channels);

        % Loop over participants and extract the relevant data
        for i = 1:num_participants
            participant_id = participant_ids{i};
            
            % Get data for movie and phone conditions for the current parameter
            movie_data(i, 1:num_incl_channels) = sw_pars.(participant_id).wave_pars_movie.(par_name);
            phone_data(i, 1:num_incl_channels) = sw_pars.(participant_id).wave_pars_phone.(par_name);
        end
        
        % Prepare electrode structure from 'chanlocs'
        elec = [];
        elec.label = {chanlocs(incl_channels).labels};  % Cell array of channel labels
        elec.pnt = [[chanlocs(incl_channels).X]', [chanlocs(incl_channels).Y]', [chanlocs(incl_channels).Z]'];  % Electrode positions

        % Create FieldTrip data structures
        data_phone = [];
        data_movie = [];
        data_phone.label = elec.label;
        data_movie.label = elec.label;
        data_phone.time = 1;  % Single dummy time point
        data_movie.time = 1;
        data_phone.dimord = 'subj_chan_time';
        data_movie.dimord = 'subj_chan_time';
        data_phone.individual = phone_data;
        data_movie.individual = movie_data;
        
        % Set up the cluster-based permutation test
        cfg = [];
        cfg.method = 'montecarlo';          % Use the Monte Carlo permutation method
        cfg.statistic = 'depsamplesT';      % Paired t-test
        cfg.correctm = 'cluster';           % Perform cluster-based correction
        cfg.clusteralpha = 0.05;            % Alpha level for forming clusters
        cfg.clusterstatistic = 'maxsum';    % Use the maximum sum of t-values for cluster comparison
        cfg.minnbchan = 2;                  % Minimum number of neighboring electrodes to form a cluster
        cfg.tail = 0;                       % Two-tailed test
        cfg.clustertail = 0;                
        cfg.alpha = 0.05;                   % Significance level
        cfg.numrandomization = 1000;        % Number of permutations
        cfg.elec = elec;                    % Include the electrode information in cfg

        % Define neighbors using the electrode layout
        cfg_neighb = [];
        cfg_neighb.method = 'distance';  % Define neighbors by distance
        cfg_neighb.elec = elec;  % Provide the electrode structure
        cfg_neighb.neighbourdist = 0.55;  % Adjust the neighbor distance threshold
        cfg.neighbours = ft_prepare_neighbours(cfg_neighb);  % Prepare neighbours
        
        % Define design (number of subjects and conditions)
        n_subjects = size(phone_data, 1);  % Number of subjects
        design = [1:n_subjects, 1:n_subjects; ones(1, n_subjects), 2 * ones(1, n_subjects)];
        cfg.design = design;
        cfg.uvar = 1;  % Subject variable
        cfg.ivar = 2;  % Independent variable (conditions)
        
        % Run the cluster-based permutation test
        stat = ft_timelockstatistics(cfg, data_phone, data_movie);
        
        % Extract significant clusters for visualization
        t_stat = stat.stat;  % Extract t-values from the FieldTrip structure
        
        % Initialize vector for channel significance
        significant = zeros(1, num_incl_channels);   % significance (yes = 1, no = 0) per channel

        % Find significant positive clusters (with p < 0.05)
        if isfield(stat, 'posclusters') && ~isempty(stat.posclusters)
            for cluster = 1:length(stat.posclusters)
                if stat.posclusters(cluster).prob < 0.05  % Check if the cluster is significant
                    % Set the value of significant electrodes to 1
                    significant(stat.posclusterslabelmat == cluster) = 1;
                end
            end
        end

        % Find significant negative clusters (with p < 0.05)
        if isfield(stat, 'negclusters') && ~isempty(stat.negclusters)
            for cluster = 1:length(stat.negclusters)
                if stat.negclusters(cluster).prob < 0.05  % Check if the cluster is significant
                    significant(stat.negclusterslabelmat == cluster) = 1;
                end
            end
        end
        
        % Find significant channels
        significant_channels = find(significant);

        % Store t-values and significant channels for this parameter
        t_vals.(par_name) = t_stat';
        sign_ch.(par_name) = significant_channels;
    end

if options.visualize
    % Create topographical plots for the t-values and highlight significant channels
    visualize_wave_pars_new_v2(t_vals, chanlocs(incl_channels), 'highlight_channels', sign_ch);
end

end