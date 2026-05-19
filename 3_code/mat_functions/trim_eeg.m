function trimmed_EEG = trim_eeg(EEG, concat_audiodir, order_file_path)
%TRIM_EEG - Trims EEG data to match variable-length audio file durations.
% Automatically adds 'boundary' markers between stitched blocks for AMICA.

triggers = {EEG.event.type};
num_triggers = length(triggers);
        
%% load matching block order file
emo_order_table = readtable(order_file_path);
cont_audio_paths = emo_order_table.audio_path;
num_blocks = height(emo_order_table);
% sanity check
if num_blocks ~= num_triggers
    error('Number of EEG triggers (%d) does not match number of rows in order file (%d).', ...
        num_triggers, num_blocks)
end

%% Initialize matrix to hold time windows
% This matrix will have 33 rows and 2 columns: [Start_Seconds, End_Seconds]
timeWindowsSec = zeros(num_blocks, 2);

% Initialize your cell array if you still need it elsewhere
cont_audio_filenames = cell(1, num_blocks);

for block_idx = 1:num_blocks
    % Get the sample points (latency) from the EEG structure
    startPoint = EEG.event(block_idx).latency;
    % Convert onset sample point to absolute time in seconds
    % Formula: (Sample Point - 1) / EEG Sampling Rate
    startTimeSec = (startPoint - 1) / EEG.srate;

    % get base name and extension
    [~, cont_name, ext] = fileparts(cont_audio_paths{block_idx});
    cont_audio_fullname = string(cont_name) + string(ext);   % e.g., "Fem_CDS_xxx.wav"
    cont_audio_filenames{1, block_idx} = cont_audio_fullname;

    % load the concatenated audio and measure the duration
    audio_file_path = fullfile(concat_audiodir, char(cont_audio_fullname));
    [audio_data, fs_audio] = audioread(audio_file_path);
    audio_duration = length(audio_data) / fs_audio; % duration in seconds

    % Calculate the end time in seconds
    endTimeSec = startTimeSec + audio_duration; 
    
    % Store the window bounds in our matrix
    timeWindowsSec(block_idx, 1) = startTimeSec;
    timeWindowsSec(block_idx, 2) = endTimeSec;
end

%% Extract windows and automatically reject the gaps
fprintf('Trimming gaps... Keeping %d audio blocks.\n', num_blocks);

% pop_select retains only the time windows specified and auto-inserts 'boundary' events
trimmed_EEG = pop_select(EEG, 'time', timeWindowsSec);

% Refresh the EEGLAB event indexing integrity
trimmed_EEG = eeg_checkset(trimmed_EEG, 'eventconsistency');
end