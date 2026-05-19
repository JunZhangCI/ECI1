clearvars; clc; close all

%% Configurations
group = 'pilot';
subs = {'sub-pilot_1'};

%% Path setup
codefolder = fileparts(fileparts(mfilename('fullpath')));
projectfolder = fileparts(codefolder);
if strcmpi(group, 'NH')
    datadir = fullfile(projectfolder, '2_data', '2_processed', 'NH');
elseif strcmpi(group, 'CI')
    datadir = fullfile(projectfolder, '2_data', '2_processed', 'CI');
elseif strcmpi(group, 'pilot')
    datadir = fullfile(projectfolder, '2_data', '2_processed', 'pilot');
else
    error('%s is not a group of the ECI1 study', group)
end
% temporary stimuli dir for pilot data
stimdir = fullfile(projectfolder, '1_studysetup', 'pilot', 'stimuli');

%% Sub selection
% if no subjects defined, automatically detect folders
if isempty(subs)
    subfolders=dir(fullfile(datadir,'sub*'));
    subfolders=subfolders([subfolders.isdir]); % keep only directories
    subs = {subfolders.name}; % keep subject names as strings
end
fprintf('\nSubjects to process:\n');
disp(subs);

%% Loop over subject
for sub = 1:length(subs)
    subname = subs{sub};
    sub_indir = fullfile(datadir, subname, 'ref_down_filt_chRej', 'ica');
    sub_outdir = fullfile(sub_indir, 'by_trial');
    if ~exist(sub_outdir,'dir')
        mkdir(sub_outdir);
    end
    sub_in_files = find_subject_files(sub_indir, 'mat', 'emo');
    for file = 1:length(sub_in_files)
        set_file_path = sub_in_files{file};
        [~, fname] = fileparts(set_file_path);
        outname = sprintf('%s_by_trial.mat', fname);
        if exist(fullfile(sub_outdir, outname), 'file')
            fprintf('Skipping %s — already processed.\n', outname);
            continue
        end
        namepts = strsplit(fname,'_');
        acqfull = namepts{5};
        acqnum = regexp(acqfull,'\d+','match','once');
        order = str2double(acqnum);
        order_excel = fullfile(stimdir, 'orders', sprintf('order%d.xlsx', order));
        order_tab = readtable(order_excel);
        trig_values = order_tab.trig_value;
        audio_paths = order_tab.audio_path;
        stim_idx = order_tab.stim_idx;
        tmp = load(set_file_path);
        EEG = tmp.EEG;

        % remove boundary events
        event_types = string({EEG.event.type});
        boundary_idx = event_types == "boundary";
        EEG.event(boundary_idx) = [];
        if exist('eeg_checkset', 'file')
            EEG = eeg_checkset(EEG, 'eventconsistency');
        end

        % Prepare output variables
        trial_data = {};
        trial_emotions = {};
        trial_speech_styles = {};
        trial_genders = {};
        trial_audio_files = {};
        trial_triggers = [];
        trig_values_str = string(trig_values);
        n_events = length(EEG.event);
        
        %% loop over trigger
        for ev = 1:n_events
            % Get event trigger
            this_type = EEG.event(ev).type;
            if isnumeric(this_type)
                this_trig = string(this_type);
            else
                this_trig = string(strtrim(char(this_type)));
            end

            % Match trigger to order file
            match_idx = find(trig_values_str == this_trig, 1);
            if isempty(match_idx)
                warning('Event %d with trigger %s was not found in trig_values. Skipping.', ...
                    ev, this_trig);
                continue
            end

            % Get audio filename
            this_audio_path = audio_paths{match_idx};
            [~, audio_name, audio_ext] = fileparts(this_audio_path);
            audio_filename = [audio_name audio_ext];

            % Parse condition labels from filename
            name_parts = strsplit(audio_name, '_');
            if length(name_parts) < 3
                warning('Audio filename %s does not have at least 3 parts. Skipping.', ...
                    audio_filename);
                continue
            end
            speaking_style = name_parts{1};
            emotion = name_parts{2};
            gender = name_parts{3};

            % Resolve audio full path
            if exist(this_audio_path, 'file')
                audio_file_fullpath = this_audio_path;
            else
                audio_file_fullpath = fullfile(stimdir, this_audio_path);
            end

            % Extract EEG trial directly from current trigger to next trigger
            start_sample = round(EEG.event(ev).latency);
            if ev < n_events
                end_sample = round(EEG.event(ev + 1).latency) - 1;
            else
                end_sample = size(EEG.data, 2);
            end

            % safe check
            if start_sample < 1
                warning('Event %d has start sample < 1. Resetting to 1.', ev);
                start_sample = 1;
            end
            if end_sample > size(EEG.data, 2)
                warning(['Trial for event %d exceeds EEG data length. ' ...
                         'Truncating end sample from %d to %d.'], ...
                         ev, end_sample, size(EEG.data, 2));
                end_sample = size(EEG.data, 2);
            end
            if end_sample < start_sample
                warning('Event %d has invalid sample window: start=%d, end=%d. Skipping.', ...
                    ev, start_sample, end_sample);
                continue
            end
            this_trial_eeg = EEG.data(:, start_sample:end_sample);

            % Save trial-level variables
            trial_data{end+1, 1} = this_trial_eeg;
            trial_emotions{end+1, 1} = emotion;
            trial_speech_styles{end+1, 1} = speaking_style;
            trial_genders{end+1, 1} = gender;
            trial_audio_files{end+1, 1} = audio_filename;
            trial_triggers(end+1, 1) = str2double(this_trig);
        end
        %% Reorder trials by stim_idx
        % Sort stim_idx in ascending order
        [stim_idx_sorted, sort_order] = sort(stim_idx, 'ascend');

        % Reorder trial-level variables
        trial_data = trial_data(sort_order);
        trial_emotions = trial_emotions(sort_order);
        trial_speech_styles = trial_speech_styles(sort_order);
        trial_genders = trial_genders(sort_order);
        trial_audio_files = trial_audio_files(sort_order);
        trial_triggers = trial_triggers(sort_order);

        %% Save output
        trial_EEG = struct();
        trial_EEG.data = trial_data;
        trial_EEG.emotions = trial_emotions;
        trial_EEG.speech_styles = trial_speech_styles;
        trial_EEG.genders = trial_genders;
        trial_EEG.audio_files = trial_audio_files;
        trial_EEG.triggers = trial_triggers;

        % meta data
        trial_EEG.srate = EEG.srate;
        trial_EEG.chanlocs = EEG.chanlocs;
        trial_EEG.subject = subname;
        trial_EEG.source_file = set_file_path;
        save_path = fullfile(sub_outdir, outname);
        save(save_path, 'trial_EEG', '-v7.3');

        fprintf('Saved trial-level data to:\n%s\n', save_path);

    end
end