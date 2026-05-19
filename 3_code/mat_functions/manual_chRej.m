function [EEG, removed_str, flat_str] = manual_chRej(EEG, fname, flat_thresh)
% MANUAL_CHREJ(EEG, logfile, fname, flat_thresh)
%   
%   This function provides a tool for user to visually inspect the
%   waveforms of EEG data by channel. Basing on the inspection and the
%   semiauto flat channel detection outcome, user can reject channels
%   producing artifacts. Rejected channels will be interpolated to preserve
%   the original channel layout. All actions would also be saved to a log
%   file.
%   
%   Key features:
%       - Extracts EEG-only channels based on EEG.chalocs.type which should
%       be already assigned ahead of this function.
%       - Automatically detects flat channels using the standard deviation
%       threshold provided.
%       - Displays channel spectra and EEG traces for visual review.
%       - User can iteratively enter channels names for removal.
%       - Replots EEG traces after each removal for confirmation.
%       - Interpolates rejected channels using spherical interpolation
%       - Saves detected flat channels and manually rejected channels to a
%       text file.
%
%   Inputs:
%     EEG         - EEGLAB EEG structure containing continuous data with
%                   channel types defined in EEG.chanlocs.type.
%
%     logfile     - Full path to a text file where flat-channel warnings and
%                   manually rejected channels will be appended.
%
%     fname       - Name of the current dataset or file (string). Used in plot
%                   titles and log entries for traceability.
%
%     flat_thresh - Threshold for flat channel detection in volts.
%                   Channels with standard deviation < flat_thresh are
%                   flagged as flat (e.g., 3e-6 corresponds to 3 µV).
%
%   Outputs:
%     EEG         - Updated EEGLAB structure after manual rejection and
%                   interpolation of selected channels.
%
%   Example usage:
%     EEG = local_manual_reject(EEG, ...
%           'C:/EEGProject/sub-001_channel_removal_log.txt', ...
%           'sub-001_task-rest', ...
%           3e-6);
%
%   Notes:
%     - Only channels with type 'EEG' are considered for rejection; EOG, REF,
%       and STIM channels are preserved.
%     - Flat channels are reported as warnings but are not automatically
%       removed; the user decides whether to reject them.
%     - If no channels are entered, the dataset is returned unchanged and
%       "n/a" is written to the log.
%     - Requires EEGLAB functions: pop_select, pop_spectopo, pop_eegplot,
%       pop_interp, eeg_checkset.

removed_str = 'none';
flat_str    = 'none';

%% Prepare dataset for channel rejection
% Split dataset into EEG-only and non-EEG for plotting
types = {EEG.chanlocs.type};
eeg_idx = find(strcmpi(types, 'EEG'));        % indices in the full EEG struct that are EEG channels
%non_eeg_idx = find(~strcmpi(types, 'EEG'));   % indices of non-EEG channels (if you want them)
% Create EEG-only dataset for plotting (keeps chanlocs consistent)
if ~isempty(eeg_idx)
    EEG_only = pop_select(EEG, 'channel', eeg_idx);
else
    error('No EEG channels found for plotting.');
end

%% Flat channel detection
flat_channels = find(std(double(EEG_only.data), 0, 2) < flat_thresh); % consider flat if standard deviation is less than 1 µV (0 means sample SD)
if ~isempty(flat_channels)
    flat_str = strjoin({EEG_only.chanlocs(flat_channels).labels}, ', ');
    warning('Flat channels detected (< %.2f µV STD): %s', ...
        flat_thresh*1e6, flat_str);
end

%% Inspect channel waveform and spetra
% eegChans = find(strcmp({EEG.chanlocs.type}, 'EEG'));
% pop_spectopo(EEG_only, 1, [], 'EEG', 'percent', 50, 'freqrange', [1 40], 'chanind', eeg_idx);
% pop_eegplot(EEG, 1, 1, 0, 1:EEG.nbchan);
pop_spectopo(EEG_only, 1, [], 'EEG', 'percent', 50, 'freqrange', [1 40], 'chanind', 1:EEG_only.nbchan);
title(['Spectral check for ' fname], 'Interpreter', 'none');
% waveform
% pop_eegplot(EEG_only, 1, 1, 0, eeg_idx, 'winlength', 60);
pop_eegplot(EEG_only, 1, 1, 0, 1:EEG_only.nbchan, 'winlength', 60);
title(['Waveform by channel for' fname], 'Interpreter', 'none');
pause;

%% Mannually select bad channels
while true
    % Ask user for channels to remove by name
    choice = questdlg( ...
    'Do you want to manually remove any channels?', ...
    'Channel Rejection', ...
    'Yes','No','No');
    if strcmp(choice,'No')
        break;
    end
    answer_cell = inputdlg( ...
        'Enter bad channel names (e.g., Fp1 P8 Cz):', ...
        'Manual Channel Rejection', ...
        [1 60]);
    if isempty(answer_cell)
        disp('User cancelled channel entry. Returning to menu...');
        continue;   % go back to questdlg instead of break
    end

    if isempty(strtrim(answer_cell{1}))
        break;
    end
    badNames = strsplit(strtrim(answer_cell{1}));

    % Keep only valid EEG channel labels
    eeg_labels = {EEG.chanlocs(eeg_idx).labels};
    existingCh = intersect(badNames, eeg_labels, 'stable');
    if isempty(existingCh)
        disp('No valid EEG channels entered.');
        continue
    end

    % Preview removal
    EEG_preview = remove_ch(EEG, existingCh, 1);
    eeg_idx2  = find(strcmpi({EEG_preview.chanlocs.type}, 'EEG'));
    EEG_only2 = pop_select(EEG_preview, 'channel', eeg_idx2);
    hEeg = pop_eegplot(EEG_only2, 1, 1, 0, 1:EEG_only2.nbchan);
    waitfor(hEeg);
    pause;
    
    % Ask if user is satisfied
    % satis = input('Are you satisfied with the channel selection? (y/n): ','s');
    satis = questdlg( ...
        'Are you satisfied with the channel selection?', ...
        'Channel rejection', ...
        'Yes', 'No', 'No');
    if strcmpi(satis,'Yes')
        % Interpolate removed channels back in using original layout
        EEG = remove_ch(EEG, existingCh, 1);
        removed_str = strjoin(existingCh, ', ');
        EEG = eeg_checkset(EEG);
        disp('Channel selection finalized and interpolated.');
        break
    else
        disp('Ok—enter a new set of channels to inspect.');
    end
end
end