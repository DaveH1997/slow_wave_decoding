% Driver to run the timelocked pipeline and produce topoplots

% 0) Load toolboxes
eeglab nogui;
ft_defaults;

% 1) Paths & params
data_path   = '/data1/s3821013';
load('chanlocs.mat');
chanlocs = chanlocs(1:62);

mode        = "tap";     % or "sw"
half_window = 1.25;      % seconds
n_bins = 25;

outdir = fullfile(data_path,'timelocked_output');
if ~exist(outdir,'dir'), mkdir(outdir); end
if ~exist(fullfile(outdir,'unsmoothed'),'dir'), mkdir(fullfile(outdir,'unsmoothed')); end
if ~exist(fullfile(outdir,'smoothed'  ),'dir'), mkdir(fullfile(outdir,'smoothed')); end

% 2) Compute peri‐event densities
results = compute_perievent_density( ...
    mode, half_window, n_bins, data_path, outdir ...
);

% 2.5) Reorder chanlocs to match the row‐order of results.params.reorder_list
chanlocs = chanlocs(results.params.reorder_list);

% 3) FieldTrip spatio-temporal cluster test (one-sample t-test against zero)
stat_us = compute_cluster_stats_ft(results, chanlocs, 'unsmoothed_z');
save(fullfile(outdir,'unsmoothed','stat_us.mat'), 'stat_us');
stat_sm = compute_cluster_stats_ft(results, chanlocs, 'smoothed_z');
save(fullfile(outdir,'smoothed','stat_sm.mat'), 'stat_sm');

% 4) Create group-level pooled z-maps with significant clusters bounded
grp_dir = fullfile(outdir,'timelocked_plots','group');
if ~exist(grp_dir,'dir'), mkdir(grp_dir); end

if mode=="tap"
    plot_xlabel = 'Time relative to tap (s)';
else
    plot_xlabel = 'Time relative to SW negative peak (s)';
end

plot_tlock_map_with_bounds( ...
    results.pooled.unsmoothed_mean, stat_us, ...
    results.params.time_centers, ...
    results.params.border_ys, ...
    plot_xlabel, ...
    fullfile(grp_dir,'pooled_mean_unsmoothed_z_bounded.svg'), ...
    'Group Mean Unsmoothed z (bounded)' ...
);

plot_tlock_map_with_bounds( ...
    results.pooled.smoothed_mean, stat_sm, ...
    results.params.time_centers, ...
    results.params.border_ys, ...
    plot_xlabel, ...
    fullfile(grp_dir,'pooled_mean_smoothed_z_bounded.svg'), ...
    'Group Mean Smoothed z (bounded)' ...
);

% 5) Create topoplots for each bin
bin_idxs = 1:n_bins;

color_u = max(abs(results.pooled.unsmoothed_mean(:)));
plot_n = 0;
for bi = bin_idxs
    plot_n = plot_n + 1;
    vals = results.pooled.unsmoothed_mean(:,bi)';
    % significant channels
    mask = zeros(size(stat_us.posclusterslabelmat));
    for c = 1:numel(stat_us.posclusters)
        if stat_us.posclusters(c).prob < 0.05
            mask(stat_us.posclusterslabelmat==c) = 1;
        end
    end
    for c = 1:numel(stat_us.negclusters)
        if stat_us.negclusters(c).prob < 0.05
            mask(stat_us.negclusterslabelmat==c) = 1;
        end
    end
    sig_ch = find(mask(:,bi));
    % plot & save
    topoplot_timelocked_density( ...
        vals, chanlocs, color_u, ...
        'plot_number', plot_n, ...
        'highlight_channels', sig_ch, ...
        'output_path', fullfile(outdir,'unsmoothed',sprintf('unsmoothed_zmap_bin%d.svg',bi)) ...
    );
end

color_s = max(abs(results.pooled.smoothed_mean(:)));
plot_n = 0;
for bi = bin_idxs
    plot_n = plot_n + 1;
    vals = results.pooled.smoothed_mean(:,bi)';
    % significant channels
    mask = zeros(size(stat_sm.posclusterslabelmat));
    for c = 1:numel(stat_sm.posclusters)
        if stat_sm.posclusters(c).prob < 0.05
            mask(stat_sm.posclusterslabelmat==c) = 1;
        end
    end
    for c = 1:numel(stat_sm.negclusters)
        if stat_sm.negclusters(c).prob < 0.05
            mask(stat_sm.negclusterslabelmat==c) = 1;
        end
    end
    sig_ch = find(mask(:,bi));
    % plot & save
    topoplot_timelocked_density( ...
        vals, chanlocs, color_s, ...
        'plot_number', plot_n, ...
        'highlight_channels', sig_ch, ...
        'output_path', fullfile(outdir,'smoothed',sprintf('smoothed_zmap_bin%d.svg',bi)) ...
    );
end

fprintf('All analyses and topoplots complete.\n');