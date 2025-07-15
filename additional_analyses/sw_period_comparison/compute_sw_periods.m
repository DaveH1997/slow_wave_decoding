%----------------------------------------------------------------------
% Compute per‐participant and group‐level slow-wave periods,
% compare movie vs. phone conditions with a paired t-test.
%----------------------------------------------------------------------

load('top10_SWs.mat'); % load summary struct

%%

% 1. Initialize
participantIDs = fieldnames(top10_SWs);
nP = numel(participantIDs);

% Preallocate per‐participant mean periods
movieMeans = nan(nP,1);
phoneMeans = nan(nP,1);

% 2. Loop through each participant
for i = 1:nP
    p = top10_SWs.(participantIDs{i});
    
    % Concatenate all channel onsets (negzx) and offsets (wvend)
    negzx = [];
    wvend = [];
    chans = p.top10_filtered_results.channels;
    
    for c = 1:numel(chans)
        % Unwrap negzx
        tmpN = chans(c).negzx;
        if iscell(tmpN)
            tmpN = cell2mat(tmpN);
        end
        negzx = [negzx; tmpN(:)];
        
        % Unwrap wvend
        tmpW = chans(c).wvend;
        if iscell(tmpW)
            tmpW = cell2mat(tmpW);
        end
        wvend = [wvend; tmpW(:)];
    end
    
    % 3. Compute durations (in ms, since Fs = 1000 Hz)
    durations = wvend - negzx;
    
    % 4. Identify waves fully within each condition
    isMovie = (negzx >= p.movie_start) & (wvend <= p.movie_end);
    isPhone = (negzx >= p.phone_start) & (wvend <= p.phone_end);
    
    % 5. Participant‐level mean period per condition
    movieMeans(i) = mean(durations(isMovie));
    phoneMeans(i) = mean(durations(isPhone));
end

% 6. Group‐level statistics
groupMovieMean  = mean(movieMeans);
groupMovieStd   = std(movieMeans);
groupPhoneMean  = mean(phoneMeans);
groupPhoneStd   = std(phoneMeans);

% 7. Paired t-test
[h, pVal, ci, stats] = ttest(movieMeans, phoneMeans);

% 8. Display results
fprintf('Movie condition: mean = %.2f ms, SD = %.2f ms\n', ...
        groupMovieMean, groupMovieStd);
fprintf('Phone condition: mean = %.2f ms, SD = %.2f ms\n', ...
        groupPhoneMean, groupPhoneStd);
fprintf('Paired t-test: t(%d) = %.2f, p = %.3f\n', ...
        stats.df, stats.tstat, pVal);