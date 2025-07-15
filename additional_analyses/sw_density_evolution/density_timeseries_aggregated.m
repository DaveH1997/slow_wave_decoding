% The channel ('ch') must be defined outside of this script in order for the plot to be saved correctly.

%% Data Preparation

% Initialize matrices for the last 40 bins of the movie condition and first 40 bins of the phone condition
num_participants = size(data_cell_array_dens, 1);
num_bins = 40; % Desired number of bins
movie_matrix = NaN(num_participants, num_bins); % Preallocate with NaNs
phone_matrix = NaN(num_participants, num_bins); % Preallocate with NaNs

% Loop through each participant and extract the required bins
for i = 1:num_participants
    % Extract movie and phone bins for the current participant
    movie_bins = data_cell_array_dens{i, 2}; % Movie condition bins
    phone_bins = data_cell_array_dens{i, 3}; % Phone condition bins
    
    % Fill the movie matrix with the last 40 bins before the movie condition ends
    if length(movie_bins) >= num_bins
        movie_matrix(i, :) = movie_bins(end-num_bins+1:end);
    else
        % Fill only the available bins if less than 40 are present
        movie_matrix(i, end-length(movie_bins)+1:end) = movie_bins;
    end
    
    % Fill the phone matrix with the first 40 bins from the start of the phone condition
    if length(phone_bins) >= num_bins
        phone_matrix(i, :) = phone_bins(1:num_bins);
    else
        % Fill only the available bins if less than 40 are present
        phone_matrix(i, 1:length(phone_bins)) = phone_bins;
    end
end

%% Aggregated time-series plot (median & median absolute deviation)

% -------------------------
% Data Preparation
% -------------------------
% Compute column medians and median absolute deviations (omitting NaNs)
movie_median = median(movie_matrix, 1, 'omitnan');   % 1x40 vector
phone_median = median(phone_matrix, 1, 'omitnan');   % 1x40 vector

movie_mad = mad(movie_matrix, 1, 1); % 1x40 vector (Scale factor set to 1)
phone_mad = mad(phone_matrix, 1, 1); % 1x40 vector (Scale factor set to 1)

% Create a time vector for plotting
movie_time = -40: -1;  % Last 40 bins of the movie condition
phone_time = 1:40;     % First 40 bins of the phone condition
time_vector = [movie_time, phone_time]; % Combined time vector

% Combine medians and MADs
combined_median = [movie_median, phone_median];   % 1x80 vector
combined_mad = [movie_mad, phone_mad];            % 1x80 vector

% Define common bin edges for histograms
edges = 0:0.5:16;

% Define common upper x-axis limits for histograms
upper_xlim_hist = 0.6;

% -------------------------
% Figure and Subplot Configuration
% -------------------------
% Create the figure
figure('Position', [100, 100, 1200, 600]);

% Define the width ratio and gaps
left_hist_width = 0.16;   % Width for left histogram
main_plot_width = 0.48;   % Width for main plot
right_hist_width = 0.16;  % Width for right histogram
gap_between = 0.03;       % Gap between subplots
total_subplot_width = left_hist_width + main_plot_width + right_hist_width; % 0.80
total_gaps = 2 * gap_between; % 0.06
total_used = total_subplot_width + total_gaps; % 0.86
remaining_width = 1 - total_used; % 0.14
left_margin = remaining_width / 2; % 0.07
right_margin = remaining_width / 2; % 0.07

% Calculate positions based on the defined widths and gaps
left_pos = [left_margin, 0.15, left_hist_width, 0.7]; % [x, y, width, height]
main_pos = [left_pos(1) + left_pos(3) + gap_between, 0.15, main_plot_width, 0.7];
right_pos = [main_pos(1) + main_pos(3) + gap_between, 0.15, right_hist_width, 0.7];

% Ensure that the right_pos does not exceed the figure width
assert(right_pos(1) + right_pos(3) + right_margin <= 1, ...
    'Subplots exceed the figure width. Adjust the margins or widths.');

% -------------------------
% Left Histogram (Movie Condition)
% -------------------------
left_axes = axes('Position', left_pos);

% Compute histogram data for movie_median (relative frequency)
[counts_movie, edges_movie] = histcounts(movie_median, edges, 'Normalization', 'probability');
bin_centers_movie = edges_movie(1:end-1) + diff(edges_movie)/2;

% Plot the histogram
barh(left_axes, bin_centers_movie, counts_movie, ...
    'FaceColor', 'b', ...
    'EdgeColor', 'b', ...
    'BarWidth', 1, ...
    'FaceAlpha', 0.2);

% Flip x-axis to extend leftwards
set(left_axes, 'XDir', 'reverse');

% Adjust axes
xlim(left_axes, [0 upper_xlim_hist]);                       % Set x-axis limits
ylim(left_axes, [0 16]);                        % Set y-axis limits to match main plot

% Add y-axis label
ylabel(left_axes, 'Median Slow-Wave Density (min^{-1})');

% Show x-axis ticks and labels for relative frequency
xlabel(left_axes, 'Relative Frequency');

% Add y-axis ticks and labels for bins
set(left_axes, 'YTick', 0:2:16, 'YTickLabel', string(0:2:16));

% Add grid
grid(left_axes, 'on');

% Remove title
title(left_axes, '');

% -------------------------
% Right Histogram (Phone Condition)
% -------------------------
right_axes = axes('Position', right_pos);

% Compute histogram data for phone_median (relative frequency)
[counts_phone, edges_phone] = histcounts(phone_median, edges, 'Normalization', 'probability');
bin_centers_phone = edges_phone(1:end-1) + diff(edges_phone)/2;

% Plot the histogram
barh(right_axes, bin_centers_phone, counts_phone, ...
    'FaceColor', 'r', ...
    'EdgeColor', 'r', ...
    'BarWidth', 1, ...
    'FaceAlpha', 0.2);

% Adjust axes
xlim(right_axes, [0 upper_xlim_hist]);                       % Set x-axis limits
ylim(right_axes, [0 16]);                        % Set y-axis limits to match main plot

% Remove redundant y-axis label
ylabel(right_axes, '');

% Set Y-Axis Location to Right and Show y-axis tick labels
set(right_axes, 'YAxisLocation', 'right');

% Show x-axis ticks and labels for relative frequency
xlabel(right_axes, 'Relative Frequency');

% Add y-axis ticks and labels for bins on the right side
set(right_axes, 'YTick', 0:2:16, 'YTickLabel', string(0:2:16));

% Add grid
grid(right_axes, 'on');

% Remove title
title(right_axes, '');

% -------------------------
% Main Plot (Time Series)
% -------------------------
main_axes = axes('Position', main_pos);
hold(main_axes, 'on');

% Plot shaded area for movie condition (±1 MAD)
patch(main_axes, [movie_time, fliplr(movie_time)], ...
      [movie_median + movie_mad, fliplr(movie_median - movie_mad)], ...
      'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Plot shaded area for phone condition (±1 MAD)
patch(main_axes, [phone_time, fliplr(phone_time)], ...
      [phone_median + phone_mad, fliplr(phone_median - phone_mad)], ...
      'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Plot the median time series
plot(main_axes, time_vector, combined_median, 'k-', 'LineWidth', 2);

% Add vertical line to mark condition change
xline(main_axes, 0, 'g--', 'LineWidth', 2, 'DisplayName', 'Condition Change');

% Adjust axes
xlim(main_axes, [-40 40]);  % Set x-axis limits to cover the time range
ylim(main_axes, [0 16]);    % Set y-axis limits to match histograms

% Add labels and title
xlabel(main_axes, 'Time (min)');
title(main_axes, 'Time-Locked Median Slow-Wave Density');

% Remove y-axis label to avoid redundancy
% ylabel(main_axes, 'Median Slow-Wave Density (min^{-1})'); % Removed as per request

% Add grid
grid(main_axes, 'on');

% Add a complete black outline to the main plot with matching thickness
set(main_axes, 'Box', 'on', ...
               'XColor', 'k', ...
               'YColor', 'k', ...
               'LineWidth', 0.5, ...
               'YTickLabel', []);

% Create legend
legend(main_axes, {'Movie ±1 MAD', 'Phone ±1 MAD', 'Median Time Series', 'Condition Change'}, ...
       'Location', 'northwest');

hold(main_axes, 'off');

% -------------------------
% Final Adjustments
% -------------------------
% Ensure y-axis limits match across all plots
set(left_axes, 'YLim', [0 16]);
set(main_axes, 'YLim', [0 16]);
set(right_axes, 'YLim', [0 16]);

% Adjust font sizes and line widths for clarity
set([left_axes, main_axes, right_axes], 'FontSize', 12);
set(findall(gcf, 'Type', 'Line'), 'LineWidth', 1.5);

%%

% Save figure
output_dir = '/data1/s3821013/sw_dens_evol_plots';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
if length(ch) == 1
    filename = sprintf('%s/sw_density_evol_ch%d.svg', output_dir, ch);
else
    filename = sprintf('%s/sw_density_evol_AVG.svg', output_dir);
end
set(gcf, 'Renderer', 'painters');
print(gcf, '-dsvg', filename);