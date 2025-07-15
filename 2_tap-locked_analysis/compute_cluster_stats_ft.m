function stat = compute_cluster_stats_ft(results, chanlocs, map_type)
% One‐sample, spatio‐temporal cluster test via FieldTrip
% stat = compute_cluster_stats_ft(results, chanlocs, map_type)

    % Number of subjects
    nsub = numel(results.participants);

    % 1) Build FT timelock struct for each subject
    D = repmat(struct('label',[],'time',[],'dimord','','avg',[]),1,nsub);
    for i = 1:nsub
        D(i).label   = { chanlocs.labels }';
        D(i).time    = results.params.time_centers;
        D(i).dimord  = 'chan_time';
        D(i).avg     = results.participants(i).(map_type);
    end

    % 2) Neighbour definition
    cfg_nb.method        = 'distance';
    cfg_nb.elec.label    = D(1).label;
    cfg_nb.elec.pnt      = [[chanlocs.X]' [chanlocs.Y]' [chanlocs.Z]'];
    cfg_nb.neighbourdist = 0.55;
    neighbours = ft_prepare_neighbours(cfg_nb);

    % 3) Cluster‐stat configuration
    cfg                  = [];
    cfg.method           = 'montecarlo';
    cfg.statistic        = 'depsamplesT';   % One-sample t-test = paired t-test: empirical data vs. vector of zeros
    cfg.parameter        = 'avg';
    cfg.correctm         = 'cluster';
    cfg.clusterstatistic = 'maxsum';
    cfg.clusteralpha     = 0.05;
    cfg.alpha            = 0.05;
    cfg.tail             = 0;
    cfg.clustertail      = 0;
    cfg.minnbchan        = 2;
    cfg.numrandomization = 1000;
    cfg.neighbours       = neighbours;

    % 4) Design: one‐sample against zero
    cfg.design = [1:nsub,    1:nsub;
                  ones(1,nsub), 2*ones(1,nsub)];
    cfg.uvar   = 1;  % row 1 = subject identifier
    cfg.ivar   = 2;  % row 2 = condition (real vs. zero)

    % 5) Build a “zero” dataset …
    D_zero = D;
    for i = 1:nsub
        D_zero(i).avg = zeros(size(D(i).avg));
    end

    % Convert both struct arrays to cell arrays of structs so we can expand them
    D_cell      = num2cell(D);
    D_zero_cell = num2cell(D_zero);

    % 6) Run the paired (real vs. zero) cluster test
    stat = ft_timelockstatistics(cfg, D_cell{:}, D_zero_cell{:});
end