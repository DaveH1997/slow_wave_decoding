function [] = topoplot_timelocked_density(density_values, chanlocs, color_limit, options)
% topoplot_timelocked_density
%   Plot a single topography of channel‐wise density values, optionally
%   highlighting a set of channels, and save directly to a specified file.
%
% Usage:
%   topoplot_timelocked_density(values, chanlocs, c_lim, ...
%       'plot_number',n, 'highlight_channels',[...], 'output_path','/path/to/file.svg')

    arguments
        density_values      double
        chanlocs            struct
        color_limit         double
        options.plot_number        double = []
        options.highlight_channels double = []
        options.output_path        char   = ''   % full filepath to save .svg
    end

    clf;
    plot_title = 'Time-Locked Analysis: Channel Z-Score';
    cb_label   = 'Z-score';

    % draw head
    if ~isempty(options.highlight_channels)
        topoplot(density_values, chanlocs, ...
            'emarker',  {'.', [0.5 0.5 0.5], 5}, ...
            'emarker2', {options.highlight_channels, '.', 'k', 15}, ...
            'style', 'both', 'shading', 'interp', 'plotrad', 0.85, 'headrad', 0.84);
    else
        topoplot(density_values, chanlocs, ...
            'emarker',  {'.', [0.5 0.5 0.5], 5}, ...
            'style', 'both', 'shading', 'interp', 'plotrad', 0.85, 'headrad', 0.84);
    end

    % custom blue-white-red colormap
    n = 256;
    b = [0 0 1]; w = [1 1 1]; r = [1 0 0];
    newmap = zeros(n,3);
    for ii = 1:floor(n/2)
        newmap(ii,:) = b + (w-b)*(ii-1)/(floor(n/2)-1);
    end
    for ii = floor(n/2)+1:n
        newmap(ii,:) = w + (r-w)*(ii-floor(n/2)-1)/(n-floor(n/2)-1);
    end
    colormap(newmap);

    clim([-color_limit, color_limit]);

    title(plot_title,'FontSize',18);
    cb = colorbar;
    cb.FontSize      = 12;
    cb.Label.String  = cb_label;
    cb.Label.FontSize= 14;
    cb.Label.Rotation= 270;

    % white face, black rim
    patch = findobj(gcf,'Type','patch');
    set(patch,'FaceColor','white','EdgeColor','black','EdgeAlpha',0);
    lines = findobj(gcf,'Type','line');

    if ~isempty(options.highlight_channels)
        set(lines(3:5),'LineWidth',1.5);
        set(lines(6),'LineWidth',3);
    else
        set(lines(2:4),'LineWidth',1.5);
        set(lines(5),'LineWidth',3);
    end

    % save
    if ~isempty(options.output_path)
        print(gcf,'-dsvg',options.output_path);
    elseif ~isempty(options.plot_number)
        fn = sprintf('timelock_topoplot%d.svg',options.plot_number);
        print(gcf,'-dsvg',fn);
    end
end