function plot_tlock_map_with_bounds( ...
    map, stat, time_centers, border_ys, ...
    plot_xlabel, out_svg, title_str)
% map            : [nchan × nbin] pooled mean z-map (already in channel-reorder order)
% stat           : FieldTrip stat struct (with .pos/negclusterslabelmat)
% time_centers   : 1×nbin vector
% border_ys      : cluster‐border y’s
% plot_xlabel    : label string
% out_svg        : full path to save .svg
% title_str      : figure title

    % 1) Identify significant cluster indices
    sigPos = find([stat.posclusters.prob] < .05);
    sigNeg = find([stat.negclusters.prob] < .05);

    % 3) Plot map
    fig = figure('Visible','off','Position',[100 100 1600 400]);
    imagesc(time_centers, 1:size(map,1), map);
    axis ij tight; hold on;

    % 4) Overlay cluster boundaries separately
    %   positive clusters
    for c = sigPos
        clustMask = stat.posclusterslabelmat == c;
        contour(time_centers, 1:size(map,1), clustMask, [0.5 0.5], ...
                'k','LineWidth',1.5);
    end
    %   negative clusters
    for c = sigNeg
        clustMask = stat.negclusterslabelmat == c;
        contour(time_centers, 1:size(map,1), clustMask, [0.5 0.5], ...
                'k-','LineWidth',1.5);
    end

    % 5) Decorations
    xlabel(plot_xlabel,'FontSize',16);
    ylabel('Channel','FontSize',16);
    title(title_str,'FontSize',18);

    % 6) Blue–white–red colormap
    n = 256;
    b = [0 0 1];  % blue
    w = [1 1 1];  % white
    r = [1 0 0];  % red
    newmap = zeros(n,3);
    for ii = 1:floor(n/2)
        newmap(ii,:) = b + (w - b) * (ii-1) / (floor(n/2)-1);
    end
    for ii = floor(n/2)+1 : n
        newmap(ii,:) = w + (r - w) * (ii - floor(n/2)-1) / (n - floor(n/2)-1);
    end
    colormap(gca, newmap);

    clim([-max(abs(map(:))) max(abs(map(:)))]);

    set(gca,'YTick',[]);
    xlims = xlim;
    for yy = border_ys
        plot(xlims, [yy yy], 'k--', 'LineWidth',1);
    end
    plot([0 0], [0.5 size(map,1)+0.5], 'k-', 'LineWidth',2);

    colorbar;

    % 7) Save & close
    print(fig,'-dsvg',out_svg);
    close(fig);
end