function trigtimes = compute_trigger_lag_emo(eeg_dir, stim_folder, newsr, debugPlotsFlag)

% Initialize trigtimes as an empty cell array%% Initialize output container
trigtimes = {};
if ~exist('debugPlotsFlag','var') || isempty(debugPlotsFlag)
    debugPlotsFlag = false;
end
%% Load EEG
EEG = pop_biosig(eeg_dir);
fs_eeg = EEG.srate;
[~, fname] = fileparts(eeg_dir);
chaninfo = struct2cell(EEG.chanlocs);
chaninfo = cellstr(char(chaninfo(1,:)));
stim_ch_idx = find(strcmpi(chaninfo,'Erg1'));
if isempty(stim_ch_idx)
    error('Erg1 channel not found in %s', fname);
end
stim_raw = EEG.data(stim_ch_idx,:);
trig_lat_samp = vertcat(EEG.event.latency);

%% extract trigger IDs from EEG.event.type
evt_raw = {EEG.event.type};
trig_codes = nan(size(evt_raw));
for i = 1:numel(evt_raw)
    val = evt_raw{i};
    while iscell(val)
        val = val{1};
    end
    if isnumeric(val)
        trig_codes(i) = val;
    else
        trig_codes(i) = str2double(val); % if it's like '1','2','11', etc.
    end

    % % debug
    % fprintf('Event %d: %s\n', i, string(val));

end

%% Preprocess stim channel
dtstim = detrend(stim_raw,1);
stim_proc = double(dtstim);
[b,a] = butter(1,1*2/fs_eeg,'high');
stim_proc = filtfilt(b,a,stim_proc);
stim_proc = stim_proc/max(abs(stim_proc));

%% Acquisition → wav sets
namepts = strsplit(fname,'_');
acqfull = namepts{5};
acqnum = regexp(acqfull,'\d+','match','once');
order = str2double(acqnum);
order_excel = fullfile(stim_folder, 'orders', sprintf('order%d.xlsx', order));
order_tab = readtable(order_excel);
audio_paths = order_tab.audio_path;
n_blocks = height(order_tab);

for block_idx = 1:n_blocks
    %% Load correct WAV

    % fprintf('%s\n', audio_paths{block_idx})
    [~, name, ext] = fileparts(audio_paths{block_idx});
    current_audio_name = [name ext];
    current_audio_path = fullfile(stim_folder, current_audio_name);
    % fprintf('%s\n', current_audio_path)
    [y, fs] = audioread(current_audio_path);

    %% Envelope + resample
    wav_env = abs(hilbert(y));
    wav_env = resample(wav_env, fs_eeg, fs);

    %% -Extract EEG segment from current trigger to next trigger (or end) 
    trig_start = round(trig_lat_samp(block_idx));   % start sample of current trigger   
    if block_idx < numel(trig_lat_samp)
        trig_end = round(trig_lat_samp(block_idx + 1)) - 1;   % sample before next trigger
    else
        trig_end = numel(stim_proc);   % last trigger -> end of file
    end
    % bounds safety
    trig_start = max(1, trig_start);
    trig_end   = min(numel(stim_proc), trig_end);
    if trig_end <= trig_start
        warning('Trigger %d has invalid segment (start=%d, end=%d). Skipping.', ...
            block_idx, trig_start, trig_end);
        continue
    end
    stim_proc_seg = stim_proc(trig_start:trig_end);

    %% Compute delay
    delay_samp = finddelay(wav_env, stim_proc_seg);

    %% Optional Debug Plot
    while debugPlotsFlag
        debugPlotsFlag = check_triggerLag_plot(wav_env, stim_proc_seg, delay_samp, 0, block_idx);
    end

    %% Save outcomes
    orig_latency_sec = trig_lat_samp(block_idx)/fs_eeg;
    delay_sec = delay_samp / fs_eeg;
    samps = round(delay_sec * newsr);
    trigtimes = [trigtimes; {fname, trig_codes(block_idx), orig_latency_sec, delay_sec, samps}];
end

end