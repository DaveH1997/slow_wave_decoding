function [preprocessed_EEG] = preprocessing_complete(EEG, varargin)

% Arguments:
% 'EEG' (struct array holding the EEG data)
% 'data_part' (data part selection [optional]; 'movie'/'phone'; defaults to using complete EEG data)
% 'ref_electrodes' (reference electrode selection [optional]; numeric vector; defaults to mastoid electrodes [52 58])

%% Handling of inputs for optional arguments

% Initialize default for 'ref_electrodes'
ref_electrodes = [52 58];

% Parse varargin for 'data_part' and 'ref_electrodes'
if ~isempty(varargin)
    for i = 1:length(varargin)
        if strcmpi(varargin{i}, 'data_part')
            data_part = varargin{i+1};
        elseif strcmpi(varargin{i}, 'ref_electrodes')
            ref_electrodes = varargin{i+1};
        end
    end
end

% Select EEG data based on 'data_part'
if exist('data_part', 'var')
    switch data_part
        case 'movie'
            EEG = pop_select(EEG, 'point', [1 4231882]); % Select movie part of EEG data
        case 'phone'
            EEG = pop_select(EEG, 'point', [4231882 7101108]); % % Select phone part of EEG data
    end
end

%% Band-pass filtering (0.5–48 Hz)

% Apply the filter
EEG = pop_eegfiltnew(EEG, 'locutoff', 0.5, 'hicutoff', 48);

%% Segmenting continuous EEG data into 4-s epochs

% % Define epoch length in seconds
% epoch_length = 4;
% 
% % Calculate the number of samples per epoch
% samples_per_epoch = epoch_length * EEG.srate;
% 
% % Number of epochs and dummy events
% number_of_epochs = floor(size(EEG.data, 2) / samples_per_epoch);
% 
% % Add dummy events to EEG structure
% for i = 1:number_of_epochs
%     EEG.event(end+1).type = 'Dummy';
%     EEG.event(end).latency = (i-1) * samples_per_epoch + 1;
%     EEG.event(end).duration = 0;
% end
% 
% % Sort the events based on latency
% [~, sortIdx] = sort([EEG.event.latency]);
% EEG.event = EEG.event(sortIdx);
% 
% % Update EEG.urevent
% EEG = eeg_checkset(EEG, 'eventconsistency');
% 
% % Epoch the data around 'Dummy' events, from 0 to 4 seconds relative to each dummy event
% EEG = pop_epoch(EEG, {'Dummy'}, [0 epoch_length]);

%% Down-sampling to 128 Hz

% Down-sample the data
EEG = pop_resample(EEG, 128);

%% Re-referencing to the average of all channels

% Re-reference the EEG data to the average of all channels
EEG = pop_reref(EEG, []);

% Save the EEG for power spectral analysis
save('rdy4spectr_analysis.mat', 'EEG');

%% Re-referencing to the average of the mastoid electrodes

% Re-reference the EEG data to the average of the left and right mastoid electrodes
EEG = pop_reref(EEG, [52 58], 'keepref', 'on');

%% Low-pass filtering (< 4 Hz)

% Apply the filter
EEG = pop_eegfiltnew(EEG, 'locutoff', [], 'hicutoff', 4);

%%

preprocessed_EEG = EEG;

end