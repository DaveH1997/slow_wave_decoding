function [] = visualize_wave_pars(wave_pars, chan_locs)

% Ensure the 'Topoplots' directory exists
if ~exist('Topoplots', 'dir')
    mkdir('Topoplots');
end

dir = 'Topoplots';

field_names = fieldnames(wave_pars);

for i = 1:length(field_names)
    clf;
    field_name = field_names{i};
    if strcmp(field_name, 'wvspermin')
        plot_title = 'Slow-Wave Density';
        cb_label = 'min^-^1';
        par_name = 'density';
    elseif strcmp(field_name, 'p2pamp')
        plot_title = 'Peak-To-Peak Amplitude';
        cb_label = 'μV';
        par_name = 'p2pamp';
    elseif strcmp(field_name, 'dslope')
        plot_title = 'Downward Slope';
        cb_label = 'μV.s^-^1';
        par_name = 'dslope';
    else
        plot_title = 'Upward Slope';
        cb_label = 'μV.s^-^1';
        par_name = 'uslope';
    end
  
    topoplot(wave_pars.(field_name), chan_locs, 'style', 'both', 'shading', 'interp', 'plotrad', 0.85, 'headrad', 0.84);
    colormap(parula);
    clim([floor(min(wave_pars.(field_name))) ceil(max(wave_pars.(field_name)))]);
    title(plot_title,'FontSize', 18);
    cb = colorbar;
    cb.FontSize = 12;
    cb.Label.String = cb_label;
    cb.Label.FontSize = 18;
    cb.Label.Rotation = 270;
    cb.Label.Position = [4.5 cb.Label.Position(2) cb.Label.Position(3)];
    patch = findobj(gcf, 'Type', 'patch');
    set(patch, 'FaceColor', 'white', 'EdgeColor', 'black', 'EdgeAlpha', 0);
    lines = findobj(gcf, 'Type', 'line');
    set(lines(5), 'LineWidth', 3);
    set(lines(2:4), 'LineWidth', 1.5);
    set(lines(1), 'MarkerSize', 5);
    filename = sprintf('%s/topoplot_%s.png', dir, par_name);
    saveas(gcf, filename);
end

end