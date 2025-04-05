function [] = visualize_wave_pars_new_v2(wave_pars, chanlocs, exp_condition, current_path, options)

arguments
    wave_pars struct;
    chanlocs struct;
    exp_condition char = '';
    current_path char = fullfile(pwd, 'Topoplots');
    options.clim_density double = [];
    options.clim_p2pamp double = [];
    options.clim_dslope double = [];
    options.clim_uslope double = [];
    options.highlight_channels struct = struct();
end

switch exp_condition
    case 'm'
        target_dir_path = fullfile(current_path, 'Movie');
        if ~exist(target_dir_path, 'dir')
            mkdir(target_dir_path);
        end
    case 'p'
        target_dir_path = fullfile(current_path, 'Phone');
        if ~exist(target_dir_path, 'dir')
            mkdir(target_dir_path);
        end
    case 'o'
        target_dir_path = fullfile(current_path, 'Overall');
        if ~exist(target_dir_path, 'dir')
            mkdir(target_dir_path);
        end
    case ''
        target_dir_path = current_path;
        if ~exist(target_dir_path, 'dir')
            mkdir(target_dir_path);
        end
end

field_names = fieldnames(wave_pars);

if strcmp(exp_condition, '')

    for i = 1:length(field_names)
        clf;
        field_name = field_names{i};
        if strcmp(field_name, 'wvspermin')
            plot_title = 'Slow-Wave Density';
            cb_label = 't-value';
            par_name = 'density';
        elseif strcmp(field_name, 'p2pamp')
            plot_title = 'Peak-To-Peak Amplitude';
            cb_label = 't-value';
            par_name = 'p2pamp';
        elseif strcmp(field_name, 'dslope')
            plot_title = 'Downward Slope';
            cb_label = 't-value';
            par_name = 'dslope';
        elseif strcmp(field_name, 'uslope')
            plot_title = 'Upward Slope';
            cb_label = 't-value';
            par_name = 'uslope';
        end

        topoplot(wave_pars.(field_name), chanlocs, 'emarker', {'.', [0.5, 0.5, 0.5], 5}, 'emarker2', {options.highlight_channels.(field_name), '.', 'k', 15}, 'style', 'both', 'shading', 'interp', 'plotrad', 0.85, 'headrad', 0.84);

        %%%%%% Create custom colormap %%%%%%

        % Number of colors in the colormap
        n = 256; 

        % Define the blue to white to red transition
        b = [0, 0, 1]; % Blue
        w = [1, 1, 1]; % White
        r = [1, 0, 0]; % Red

        % Create the colormap
        newmap = zeros(n, 3);

        % Blue to white transition
        for i = 1:floor(n/2)
            newmap(i, :) = b + (w - b) * (i-1) / (floor(n/2) - 1);
        end

        % White to red transition
        for i = floor(n/2)+1:n
            newmap(i, :) = w + (r - w) * (i - floor(n/2) - 1) / (n - floor(n/2) - 1);
        end

        % Apply the colormap
        colormap(newmap);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if abs(floor(min(wave_pars.(field_name)))) > abs(ceil(max(wave_pars.(field_name))))
            clim([floor(min(wave_pars.(field_name))), abs(floor(min(wave_pars.(field_name))))]);
        else
            clim([(-ceil(max(wave_pars.(field_name)))), ceil(max(wave_pars.(field_name)))]);
        end

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

        if length(lines) == 6  % if there are sign. clusters, there will be an additional object for channel highlighters
            set(lines(6), 'LineWidth', 3); % rim
            set(lines(3:5), 'LineWidth', 1.5); % ears and nose
        else
            set(lines(5), 'LineWidth', 3); % rim
            set(lines(2:4), 'LineWidth', 1.5); % ears and nose
        end

        % filename = sprintf('%s/topoplot_%s.png', target_dir_path, par_name);
        % saveas(gcf, filename);

        filename = sprintf('%s/topoplot_%s.svg', target_dir_path, par_name);
        print(gcf, '-dsvg', filename);
    end

else

    for i = 1:length(field_names)
        clf;
        field_name = field_names{i};
        if strcmp(field_name, 'wvspermin')
            plot_title = 'Slow-Wave Density';
            cb_label = 'min^-^1';
            par_name = 'density';
            if isempty(options.clim_density)
                limits = [floor(min(wave_pars.(field_name))), ceil(max(wave_pars.(field_name)))];
            else
                limits = options.clim_density;
            end
        elseif strcmp(field_name, 'p2pamp')
            plot_title = 'Peak-To-Peak Amplitude';
            cb_label = 'μV';
            par_name = 'p2pamp';
            if isempty(options.clim_p2pamp)
                limits = [floor(min(wave_pars.(field_name))), ceil(max(wave_pars.(field_name)))];
            else
                limits = options.clim_p2pamp;
            end
        elseif strcmp(field_name, 'dslope')
            plot_title = 'Downward Slope';
            cb_label = 'μV.s^-^1';
            par_name = 'dslope';
            if isempty(options.clim_dslope)
                limits = [floor(min(wave_pars.(field_name))), ceil(max(wave_pars.(field_name)))];
            else
                limits = options.clim_dslope;
            end
        else
            plot_title = 'Upward Slope';
            cb_label = 'μV.s^-^1';
            par_name = 'uslope';
            if isempty(options.clim_uslope)
                limits = [floor(min(wave_pars.(field_name))), ceil(max(wave_pars.(field_name)))];
            else
                limits = options.clim_uslope;
            end
        end
      
        topoplot(wave_pars.(field_name), chanlocs, 'style', 'both', 'shading', 'interp', 'plotrad', 0.85, 'headrad', 0.84);
        colormap(parula);
        clim(limits);
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
        set(lines(5), 'LineWidth', 3); % rim
        set(lines(2:4), 'LineWidth', 1.5); % ears and nose
        set(lines(1), 'MarkerSize', 5); % channel markers

        % filename = sprintf('%s/topoplot_%s.png', target_dir_path, par_name);
        % saveas(gcf, filename);

        filename = sprintf('%s/topoplot_%s.svg', target_dir_path, par_name);
        print(gcf, '-dsvg', filename);
    end

end