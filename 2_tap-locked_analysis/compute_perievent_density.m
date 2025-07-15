function results = compute_perievent_density(tlock_event, half_window, n_bins, data_path, target_path)
% run_timelocked_analysis Compute and save peri‐event density analyses for participants.
%
% Usage:
%   results = run_timelocked_analysis_v2(tlock_event, half_window, data_path, target_path)
%
% Arguments:
%   tlock_event – 'tap' or 'sw', specifies time-locking event
%   half_window – half-width (in seconds) of the peri-event window (must be > 0)
%   data_path   – path to folder containing 'top10_SWs.mat'
%   target_path – base directory for outputs (plots and results)

    arguments
        tlock_event  (1,1) string {mustBeMember(tlock_event,["tap","sw"])}
        half_window  (1,1) double {mustBePositive}
        n_bins       (1,1) double {mustBePositive}
        data_path    (1,1) string
        target_path  (1,1) string
    end

    %% 1. Dynamic parameters
    fs = 1000;                             % sampling rate (Hz)
    total_window = 2 * half_window;        % full window (s)
    bin_width = total_window / n_bins;     % bin width (s)
    trim_seconds = total_window;           % trim margin (s)
    sigma_sec = 3 * bin_width;             % Gaussian σ (s)
    sigma_bins = sigma_sec / bin_width;    % σ in bins
    half_win_bins = ceil(3 * sigma_bins);  % ±3σ in bins
    ext_margin = half_win_bins * bin_width;

    t_win = [-half_window, half_window];
    ext_win = [t_win(1)-ext_margin, t_win(2)+ext_margin];
    ext_edges   = ext_win(1):bin_width:ext_win(2);
    ext_centers = ext_edges(1:end-1) + bin_width/2;
    time_edges   = t_win(1):bin_width:t_win(2);
    time_centers = time_edges(1:end-1) + bin_width/2;
    n_bins_ext = numel(ext_centers);

    % Store common parameters for downstream permutation testing
    results.params.bin_width     = bin_width;
    results.params.time_centers  = time_centers;
    results.params.half_win_bins = half_win_bins;
    results.params.reorder_list  = [];    % will set after reorder_list defined
    results.params.border_ys     = [];    % will set after border_ys defined
    results.params.g             = [];    % will set after g defined
    results.params.target_path   = target_path;

    %% 2. Determine x-axis label
    if tlock_event == "tap"
        plot_xlabel = 'Time relative to tap (s)';
    else
        plot_xlabel = 'Time relative to SW negative peak (s)';
    end

    %% 3. Load data and participants
    S = load(fullfile(data_path,'top10_SWs.mat'),'top10_SWs');
    top10_SWs = S.top10_SWs;
    participant_ids = fieldnames(top10_SWs);

    %% 4. Electrode reordering and cluster borders
    reorder_list = [49,60,48,32,47,59,34,19,33,20,7,18,8,35,50,36,21,37,51, ...
                    17,31,16,46,30,15,1,6,2,5,3,4,9,22,10,38,23,11,58,29,45,44,57,61, ...
                    14,12,13,28,25,27,26,52,24,39,40,53,62,43,41,42,56,54,55];
    n_channels = numel(reorder_list);
    cluster_sizes = [6,7,6,6,6,6,6,7,6,6];
    border_ys = cumsum(cluster_sizes); 
    border_ys = border_ys(1:end-1) + 0.5;

    % Now fill previously empty params
    results.params.reorder_list = reorder_list;
    results.params.border_ys     = border_ys;

    %% 5. Prepare smoothing kernel
    x = -half_win_bins:half_win_bins;
    g = exp(-(x.^2)/(2*sigma_bins^2));
    g = g / sum(g);

    results.params.g = g;

    %% 6. Initialize results struct
    results.participants = repmat(struct( ...
        'id','', ...
        'taps_all',[], ...
        'safe_taps',[], ...
        'sw_all',[], ...
        'sw_safe',[], ...
        'unsmoothed_raw',[], ...
        'unsmoothed_z',[], ...
        'smoothed_raw',[], ...
        'smoothed_z',[]), numel(participant_ids),1);

    %% 7. Loop over participants
    for p = 1:numel(participant_ids)
        pid = participant_ids{p};
        results.participants(p).id = pid;

        % Load and trim taps
        taps_all = top10_SWs.(pid).taps;
        ft = taps_all(1); lt = taps_all(end);
        safe_taps = taps_all( taps_all >= ft + trim_seconds*fs & ...
                              taps_all <= lt - trim_seconds*fs );

        % Store raw and trimmed event times for inference
        results.participants(p).taps_all  = taps_all;
        results.participants(p).safe_taps = safe_taps;

        % Load SW negative peak latencies per channel
        all_peaks = cell(1,64);
        for ch = 1:64
            raw = top10_SWs.(pid).top10_filtered_results.channels(ch).maxnegpk;
            all_peaks{ch} = cell2mat(raw);
        end
        % Prepare safe SW peaks
        safe_peaks = cell(1,64);
        for ch = 1:64
            pk = all_peaks{ch};
            safe_peaks{ch} = pk(pk >= ft + trim_seconds*fs & pk <= lt - trim_seconds*fs);
        end

        % Store SW peak times
        results.participants(p).sw_all  = all_peaks;
        results.participants(p).sw_safe = safe_peaks;

        % Build extended histogram
        unsm_ext = zeros(n_channels, n_bins_ext);
        for idx = 1:n_channels
            ch = reorder_list(idx);
            if tlock_event == "tap"
                events = safe_taps;      % trimmed taps
                refs   = all_peaks{ch};  % all SW peaks
            else
                events = safe_peaks{ch}; % trimmed SW peaks
                refs   = taps_all;       % all taps
            end
            rel = [];
            for e = events(:)'
                w0 = e + round(ext_win(1)*fs);
                w1 = e + round(ext_win(2)*fs);
                sel = refs >= w0 & refs <= w1;
                if any(sel)
                    rel = [rel, (refs(sel)-e)/fs]; %#ok<AGROW>
                end
            end
            if ~isempty(rel)
                unsm_ext(idx,:) = histcounts(rel, ext_edges);
            end
        end

        % Trim to nominal window and z-score (unsmoothed)
        unsm = unsm_ext(:, half_win_bins+1:end-half_win_bins);
        mu_u = mean(unsm,2); sd_u = std(unsm,0,2);
        unsm_z = (unsm - mu_u) ./ sd_u;

        % Smooth and z-score (smoothed)
        sm_ext = conv2(unsm_ext, g, 'same');
        sm = sm_ext(:, half_win_bins+1:end-half_win_bins);
        mu_s = mean(sm,2); sd_s = std(sm,0,2);
        sm_z = (sm - mu_s) ./ sd_s;

        % Store matrices
        results.participants(p).unsmoothed_raw = unsm;
        results.participants(p).unsmoothed_z   = unsm_z;
        results.participants(p).smoothed_raw   = sm;
        results.participants(p).smoothed_z     = sm_z;

        % Prepare output dirs
        base_dir = fullfile(target_path,'timelocked_plots',pid);
        if ~exist(base_dir,'dir'), mkdir(base_dir); end
        if ~exist(fullfile(base_dir,'unsmoothed'),'dir')
            mkdir(fullfile(base_dir,'unsmoothed'));
        end
        if ~exist(fullfile(base_dir,'smoothed'),'dir')
            mkdir(fullfile(base_dir,'smoothed'));
        end

        % --- Plot unsmoothed counts ---
        fig_counts = figure('Visible','off','Position',[100 100 1600 400]);
        imagesc(time_centers,1:n_channels,unsm);
        axis ij; xlabel(plot_xlabel,'FontSize',16);
        ylabel('Channel','FontSize',16);
        title('Unsmoothed Counts','FontSize',18);
        colormap(gca,parula); colorbar;
        clim([0,max(unsm(:))]);
        ylim([0.5,n_channels+0.5]);
        set(gca,'FontSize',14,'YTick',[]);
        hold on; xlims = xlim;
        for yy = border_ys, plot(xlims,[yy yy],'k--','LineWidth',1); end
        plot([0,0],[0.5,n_channels+0.5],'k-','LineWidth',3);
        hold off;
        print(fig_counts,'-dsvg',fullfile(base_dir,'unsmoothed',sprintf('unsmoothed_counts_%s.svg',pid)));
        close(fig_counts);

        % --- Plot unsmoothed z-scores ---
        fig_zscores = figure('Visible','off','Position',[100 100 1600 400]);
        imagesc(time_centers,1:n_channels,unsm_z);
        axis ij; xlabel(plot_xlabel,'FontSize',16);
        ylabel('Channel','FontSize',16);
        title('Unsmoothed Z-scores','FontSize',18);
        cmap_div = [linspace(0,1,128)' linspace(0,1,128)' ones(128,1);
                   ones(128,1) linspace(1,0,128)' linspace(1,0,128)'];
        colormap(gca,cmap_div); colorbar;
        ma = max(abs(unsm_z(:))); clim([-ma,ma]);
        ylim([0.5,n_channels+0.5]);
        set(gca,'FontSize',14,'YTick',[]);
        hold on; for yy = border_ys, plot(xlims,[yy yy],'k--','LineWidth',1); end
        plot([0,0],[0.5,n_channels+0.5],'k-','LineWidth',3);
        hold off;
        print(fig_zscores,'-dsvg',fullfile(base_dir,'unsmoothed',sprintf('unsmoothed_zscores_%s.svg',pid)));
        close(fig_zscores);

        % --- Plot smoothed counts ---
        fig_s_counts = figure('Visible','off','Position',[100 100 1600 400]);
        imagesc(time_centers,1:n_channels,sm);
        axis ij; xlabel(plot_xlabel,'FontSize',16);
        ylabel('Channel','FontSize',16);
        title('Smoothed Counts','FontSize',18);
        colormap(gca,parula); colorbar;
        clim([0,max(sm(:))]);
        ylim([0.5,n_channels+0.5]);
        set(gca,'FontSize',14,'YTick',[]);
        hold on; xlims = xlim;
        for yy = border_ys, plot(xlims,[yy yy],'k--','LineWidth',1); end
        plot([0,0],[0.5,n_channels+0.5],'k-','LineWidth',3);
        hold off;
        print(fig_s_counts,'-dsvg',fullfile(base_dir,'smoothed',sprintf('smoothed_counts_%s.svg',pid)));
        close(fig_s_counts);

        % --- Plot smoothed z-scores ---
        fig_s_zscores = figure('Visible','off','Position',[100 100 1600 400]);
        imagesc(time_centers,1:n_channels,sm_z);
        axis ij; xlabel(plot_xlabel,'FontSize',16);
        ylabel('Channel','FontSize',16);
        title('Smoothed Z-scores','FontSize',18);
        colormap(gca,cmap_div); colorbar;
        ma2 = max(abs(sm_z(:))); clim([-ma2,ma2]);
        ylim([0.5,n_channels+0.5]);
        set(gca,'FontSize',14,'YTick',[]);
        hold on; for yy = border_ys, plot(xlims,[yy yy],'k--','LineWidth',1); end
        plot([0,0],[0.5,n_channels+0.5],'k-','LineWidth',3);
        hold off;
        print(fig_s_zscores,'-dsvg',fullfile(base_dir,'smoothed',sprintf('smoothed_zscores_%s.svg',pid)));
        close(fig_s_zscores);
    end

    %% 8. Pool across participants
    Z_unsm = cat(3, results.participants.unsmoothed_z);
    Z_sm   = cat(3, results.participants.smoothed_z);
    results.pooled.unsmoothed_mean   = mean(Z_unsm,3);
    results.pooled.unsmoothed_median = median(Z_unsm,3);
    results.pooled.smoothed_mean     = mean(Z_sm,3);
    results.pooled.smoothed_median   = median(Z_sm,3);

    %% 9. Create & save group-level pooled plots
    grp_dir = fullfile(target_path,'timelocked_plots','group');
    if ~exist(grp_dir,'dir'), mkdir(grp_dir); end

    % Group mean unsmoothed z-scores
    fig_g_u_mean = figure('Visible','off','Position',[100 100 1600 400]);
    imagesc(time_centers,1:n_channels,results.pooled.unsmoothed_mean);
    axis ij; xlabel(plot_xlabel,'FontSize',16); ylabel('Channel','FontSize',16);
    title('Group Mean Unsmoothed Z','FontSize',18);
    colormap(gca,cmap_div); colorbar;
    ma3 = max(abs(results.pooled.unsmoothed_mean(:))); clim([-ma3,ma3]);
    ylim([0.5,n_channels+0.5]); set(gca,'FontSize',14,'YTick',[]);
    hold on; xlims = xlim;
    for yy=border_ys, plot(xlims,[yy yy],'k--','LineWidth',1); end
    plot([0,0],[0.5,n_channels+0.5],'k-','LineWidth',3); hold off;
    print(fig_g_u_mean,'-dsvg',fullfile(grp_dir,'pooled_mean_unsmoothed_z.svg'));
    close(fig_g_u_mean);

    % Group mean smoothed z-scores
    fig_g_s_mean = figure('Visible','off','Position',[100 100 1600 400]);
    imagesc(time_centers,1:n_channels,results.pooled.smoothed_mean);
    axis ij; xlabel(plot_xlabel,'FontSize',16); ylabel('Channel','FontSize',16);
    title('Group Mean Smoothed Z','FontSize',18);
    colormap(gca,cmap_div); colorbar;
    ma4 = max(abs(results.pooled.smoothed_mean(:))); clim([-ma4,ma4]);
    ylim([0.5,n_channels+0.5]); set(gca,'FontSize',14,'YTick',[]);
    hold on; for yy=border_ys, plot(xlims,[yy yy],'k--','LineWidth',1); end
    plot([0,0],[0.5,n_channels+0.5],'k-','LineWidth',3); hold off;
    print(fig_g_s_mean,'-dsvg',fullfile(grp_dir,'pooled_mean_smoothed_z.svg'));
    close(fig_g_s_mean);

    % Group median unsmoothed z-scores
    fig_g_u_med = figure('Visible','off','Position',[100 100 1600 400]);
    imagesc(time_centers,1:n_channels,results.pooled.unsmoothed_median);
    axis ij; xlabel(plot_xlabel,'FontSize',16); ylabel('Channel','FontSize',16);
    title('Group Median Unsmoothed Z','FontSize',18);
    colormap(gca,cmap_div); colorbar;
    ma5 = max(abs(results.pooled.unsmoothed_median(:))); clim([-ma5,ma5]);
    ylim([0.5,n_channels+0.5]); set(gca,'FontSize',14,'YTick',[]);
    hold on; xlims = xlim;
    for yy=border_ys, plot(xlims,[yy yy],'k--','LineWidth',1); end
    plot([0,0],[0.5,n_channels+0.5],'k-','LineWidth',3); hold off;
    print(fig_g_u_med,'-dsvg',fullfile(grp_dir,'pooled_median_unsmoothed_z.svg'));
    close(fig_g_u_med);

    % Group median smoothed z-scores
    fig_g_s_med = figure('Visible','off','Position',[100 100 1600 400]);
    imagesc(time_centers,1:n_channels,results.pooled.smoothed_median);
    axis ij; xlabel(plot_xlabel,'FontSize',16); ylabel('Channel','FontSize',16);
    title('Group Median Smoothed Z','FontSize',18);
    colormap(gca,cmap_div); colorbar;
    ma6 = max(abs(results.pooled.smoothed_median(:))); clim([-ma6,ma6]);
    ylim([0.5,n_channels+0.5]); set(gca,'FontSize',14,'YTick',[]);
    hold on; for yy=border_ys, plot(xlims,[yy yy],'k--','LineWidth',1); end
    plot([0,0],[0.5,n_channels+0.5],'k-','LineWidth',3); hold off;
    print(fig_g_s_med,'-dsvg',fullfile(grp_dir,'pooled_median_smoothed_z.svg'));
    close(fig_g_s_med);

    %% 10. Save results struct
    save(fullfile(target_path,'timelocked_results.mat'),'results','-v7.3');
end