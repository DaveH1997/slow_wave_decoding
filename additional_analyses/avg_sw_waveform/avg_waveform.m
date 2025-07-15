% eeglab nogui;

% % Load the raw EEG data (1000 Hz)
% EEG = pop_loadset('filename', '12_57_07_05_18.set', 'filepath', '/Users/davidhof/Desktop/MSc/3rd Semester/Internship/Local Sleep Project/Test Data/AT08');

% Load the preprocessed EEG data (128 Hz)
load('/Users/davidhof/Desktop/MSc/3rd Semester/Internship/Local Sleep Project/Test Data/AT08/preprocessed_EEG.mat');

% Load the SW characteristics (latency in ms; voltage in μV)
load('/Users/davidhof/Desktop/MSc/3rd Semester/Internship/Local Sleep Project/Test Data/AT08/top10_filtered_results.mat');

% Define the channel
ch = 1;

%%

movie_end = 4231882; % in ms

% Define window start times
window_starts = cell2mat(top10_filtered_results.channels(ch).negzx) - 500;

% % Separate window start times
% window_starts_movie = window_starts(window_starts <= movie_end);
% window_starts_phone = window_starts(window_starts > movie_end);

% % Sampling rate of the raw/preprocessed EEG data
% fs = 1000; % raw
fs = 128; % preprocessed

% Convert window start times from ms to indices (for preprocessed EEG data)
window_starts_indices = round(window_starts * fs / 1000);
% window_starts_movie_indices = round(window_starts_movie * fs / 1000);
% window_starts_phone_indices = round(window_starts_phone * fs / 1000);

% Extract the channel data
% ch_data = EEG.data(ch,:);
ch_data = preprocessed_EEG.data(ch,:);

% Initialize a matrix to store the waveforms of each window (for entire recording)
% num_windows = length(window_starts);
num_windows = length(window_starts_indices);
window_length = round(1.5 * fs);
waveforms = zeros(num_windows, window_length);

% % Initialize matrices to store the waveforms of each window in the movie and phone conditions
% % num_windows_movie = length(window_starts_movie);
% % num_windows_phone = length(window_starts_phone);
% num_windows_movie = length(window_starts_movie_indices);
% num_windows_phone = length(window_starts_phone_indices);
% window_length = round(1.5 * fs);
% waveforms_movie = zeros(num_windows_movie, window_length);
% waveforms_phone = zeros(num_windows_phone, window_length);

% Loop through each window and extract the data (for entire recording)
for i = 1:num_windows
    % waveforms(i,:) = ch_data(window_starts(i):window_starts(i) + window_length - 1);
    waveforms(i,:) = ch_data(window_starts_indices(i):window_starts_indices(i) + window_length - 1);
end

% % Loop through each window and extract the data (for movie condition)
% for i = 1:num_windows_movie
%     % waveforms_movie(i,:) = ch_data(window_starts_movie(i):window_starts_movie(i) + window_length_movie - 1);
%     waveforms_movie(i,:) = ch_data(window_starts_movie_indices(i):window_starts_movie_indices(i) + window_length - 1);
% end
% 
% % Loop through each window and extract the data (for phone condition)
% for i = 1:num_windows_phone
%     % waveforms_phone(i,:) = ch_data(window_starts_phone(i):window_starts_phone(i) + window_length_phone - 1);
%     waveforms_phone(i,:) = ch_data(window_starts_phone_indices(i):window_starts_phone_indices(i) + window_length - 1);
% end

% Calculate the average waveform across all windows (for entire recording)
average_waveform = mean(waveforms, 1);

% % Calculate the average waveforms across all windows in the movie and phone conditions
% average_waveform_movie = mean(waveforms_movie, 1);
% average_waveform_phone = mean(waveforms_phone, 1);

% Create a time vector for the x-axis from -0.5 seconds to 1 second
time_vector = linspace(-0.5, 1, window_length);

fig_title = sprintf('Average SW Waveform for Channel %d', ch);
% fig_title = sprintf('Average SW Waveforms for Channel %d', ch);

% Plot the average waveform (for entire recording)
figure;
plot(time_vector, average_waveform, 'r', 'LineWidth', 3);
xlabel('Time (s)');
ylabel('Voltage (μV)');
title(fig_title);

% % Plot the average waveforms for the movie and phone conditions
% figure;
% hold on;
% plot(time_vector, average_waveform_movie, 'r', 'LineWidth', 3);  % Plot movie waveform in red
% plot(time_vector, average_waveform_phone, 'b', 'LineWidth', 3);  % Plot phone waveform in blue
% hold off;

% Customize the axes labels and title
xlabel('Time (s)', 'FontSize', 36);
ylabel('Voltage (μV)', 'FontSize', 36);
% title(fig_title, 'FontSize', 36);
title(fig_title, 'FontSize', 36);

% Increase the font size of the tick labels
set(gca, 'FontSize', 28);

% Remove the top and right axes
set(gca, 'box', 'off');

% % Add a legend to distinguish between the two waveforms (movie and phone)
% leg_movie = sprintf('Movie (n = %d)', num_windows_movie);
% leg_phone = sprintf('Phone (n = %d)', num_windows_phone);
% legend({leg_movie, leg_phone}, 'FontSize', 28);

%%

% Save figure
filename = sprintf('%s/avg_sw_waveform_ch%d.svg', pwd, ch);
set(gcf, 'Renderer', 'painters');
print(gcf, '-dsvg', filename);