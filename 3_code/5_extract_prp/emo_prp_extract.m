clearvars; clc; close all

%% Configuration
group = 'pilot';
subject_ids = {'sub-pilot_1'};
% Set EEG sampling, PRP window, and sentence timing parameters.
eeg_fs = 128; % Hz
prp_duration = 1; % sec
baseline_duration = 0.1; % sec
pre_event_dur = 0; % sec
sentence_isi = 0.5; % sec

% Toggle optional processing steps.
detrend_mode = 1;   % 0: none, 1: linear, 2: constant
baseline_correction = 0;
trial_reject = 0;
standardize = 0;

% Set RMS-based trial rejection parameters.
reject_threshold = 2;          % std
rms_reject_channels = ["AF3", "AF4", "Fz", "F3", "F4", "FC1", "FC2"];

%% Path setup
codefolder = fileparts(fileparts(mfilename('fullpath')));
projectfolder = fileparts(codefolder);
emo_audiofolder = fullfile(projectfolder, 'emo_audio');
corpusdir =  fullfile(emo_audiofolder, '3_mfa');
unique_phoneme_path = [corpusdir '/unique_phones.txt'];
% temporary stimuli dir for pilot data
emo_concatenate_audiodir = fullfile(projectfolder, '1_studysetup', 'pilot', 'stimuli');
emo_orderdir = fullfile(emo_concatenate_audiodir, 'orders');
concat_order_path = fullfile
if strcmpi(group, 'NH')
    datadir = fullfile(projectfolder, '2_data', '2_processed', 'NH');
elseif strcmpi(group, 'CI')
    datadir = fullfile(projectfolder, '2_data', '2_processed', 'CI');
elseif strcmpi(group, 'pilot')
    datadir = fullfile(projectfolder, '2_data', '2_processed', 'pilot');
else
    error('%s is not a group of the ECI1 study', group)
end

%% process input data 
% Convert PRP timing parameters from seconds to samples.
prp_duration_samples = round(prp_duration * eeg_fs);
baseline_duration_samples = round(baseline_duration * eeg_fs);
prp_window_samples = prp_duration_samples + baseline_duration_samples;

% Load the phoneme whitelist used to keep only target phonemes.
phonemes_to_include = strsplit(strtrim(fileread(unique_phoneme_path)), newline);
phonemes_to_include = erase(phonemes_to_include, char(13));   % remove \r

fprintf('Phonemes to include in data file (%d):\n', length(phonemes_to_include));
disp(phonemes_to_include)

%% subject selection
if isempty(subject_ids)
    subject_folders = dir(fullfile(datadir, 'sub*'));
    subject_folders = subject_folders([subject_folders.isdir]);
    subject_ids = {subject_folders.name};
end

fprintf('\nSubjects to process:\n');
disp(subject_ids);

%% loop over subjects
for sub_idx = 1:length(subject_ids)
    subject_id = subject_ids{sub_idx};
    fprintf('\n---- Processing: %s ----\n', string(subject_id));
    
    % Choose the input EEG folder based on the preprocessing branch used.
    if epoch_first
        subject_input_dir = fullfile(datadir, subject_id, 'ref_down_filt_chRej/epoch_reject/ica/MAT');
    else
        subject_input_dir = fullfile(datadir, subject_id, 'ref_down_filt_chRej/ica/epoch_reject/MAT');
    end
    
    % Create output folder for PRP files if it does not exist.
    subject_output_dir = fullfile(subject_input_dir, 'prp');
    if ~exist(subject_output_dir, 'dir')
        mkdir(subject_output_dir);
    end
    
    % Find all emotion-task EEG files for this subject.
    subject_input_files = find_subject_files(subject_input_dir, 'mat', 'emo');

    %% loop over input EEG files
    for file_idx = 1:length(subject_input_files)
        mat_file_path = subject_input_files{file_idx};
        [~, input_file_name] = fileparts(mat_file_path);
        
        % Parse identifiers from the EEG filename for output naming.
        file_parts = split(input_file_name, "_");
        if epoch_first
            ica_label = char(file_parts(end));
        else
            ica_label = char(file_parts(end-2));
        end

        sub_str  = regexp(input_file_name, 'sub-pilot_[^_]+', 'match', 'once');
        ses_str  = regexp(input_file_name, 'ses-[^_]+', 'match', 'once');
        task_str = regexp(input_file_name, 'task-[^_]+', 'match', 'once');
        acq_str  = regexp(input_file_name, 'acq-[^_]+', 'match', 'once');
        
        % Build output suffix to record optional processing steps.
        output_suffix = "";
        if detrend_mode == 1
            output_suffix = output_suffix + "_detrendLinear";
        elseif detrend_mode == 2
            output_suffix = output_suffix + "_detrendConst";
        end
        if standardize == 1
            output_suffix = output_suffix + "_stand";
        end
        if trial_reject == 1
            output_suffix = output_suffix + "_rej";
        end
        
        % Create the final output filename for this EEG file.
        output_file_name = sprintf('%s_%s_%s_%s_%s_prp%s.mat', ...
            sub_str, ses_str, task_str, acq_str, ica_label, output_suffix);
        
        % Skip file if PRP output already exists.
        if exist(fullfile(subject_output_dir, output_file_name), 'file')
            fprintf('Skipping %s — already processed.\n', output_file_name);
            continue
        end

        fprintf('---- Loading: %s ----\n', input_file_name);

        %% load EEG file
        % Load EEGLAB data and channel labels from the input .mat file.
        mat_struct = load(mat_file_path);
        EEG_struct = mat_struct.EEG;
        eeg_data = EEG_struct.data;                      % [chan x time x block]
        channel_labels = {EEG_struct.chanlocs.labels};
        %preprocess_steps = EEG_struct.preprocess;

        %% load matching block order file
        % Load the stimulus order table that maps each EEG block to a stimulus file.
        order_num = str2double(regexp(input_file_name, 'acq-(\d+)', 'tokens', 'once'));
        order_file_path = fullfile(emo_orderdir, sprintf('order%d.xlsx', order_num));

        if exist(order_file_path, 'file')
            block_order_table = readtable(order_file_path);
        else
            fprintf('Order file not found for %s.\n', input_file_name);
            continue
        end

        block_audio_paths = block_order_table.audio_path;
        num_blocks = height(block_order_table);

        %% initialize outputs
        % Initialize PRP storage arrays and extraction counters.
        prp_trials_eeg = [];             % [chan x time x nPRP]
        prp_trim_begin = 0;
        prp_trim_end = 0;
        prp_invalid_window = 0;
        prp_unmatched_duration = 0;

        prp_phoneme_labels = {};
        prp_onset_times_epoch = [];
        prp_emotions = {};
        prp_speech_styles = {};
        prp_genders = {};

        %% loop over blocks
        for block_idx = 1:num_blocks
            % Get block audio name and recover metadata encoded in the filename.
            [~, block_audio_stem_raw, audio_ext] = fileparts(block_audio_paths{block_idx});
            block_audio_stem = erase(string(block_audio_stem_raw), "_scaled");
            block_audio_filename = string(block_audio_stem) + string(audio_ext);
            
            name_parts = split(string(block_audio_stem), "_");
            current_gender = name_parts(1);
            current_speech_style = name_parts(2);
            current_emotion = name_parts(3);
            
            % Find the sentence order table for the current speaker/style block.
            sentence_order_dir = fullfile(concat_audiodir, current_speech_style, current_gender + "_" + current_speech_style);
            sentence_order_name = sprintf('%s_%s_random_cont_order.csv', current_gender, current_speech_style);
            sentence_order_path = fullfile(sentence_order_dir, sentence_order_name);
            
            % Recover the ordered list of sentence IDs used in this block.
            if isfile(sentence_order_path)
                sentence_order_list = getSentenceOrderFromTable(sentence_order_path, block_audio_filename);
            else
                fprintf('Sentence order table not found for %s.\n', sentence_order_name);
                continue
            end
            
            % Select the EEG epoch corresponding to this stimulus block.
            block_eeg = eeg_data(:, :, block_idx);   % [chan x time]
            sentence_start_time_epoch = pre_event_dur;

            %% loop over sentences
            for sentence_idx = 1:length(sentence_order_list)
                % Build paths to the sentence TextGrid and audio files.
                sentence_id = sentence_order_list{sentence_idx};

                sentence_textgrid_name = sprintf('%s_cleaned.TextGrid', sentence_id);
                sentence_textgrid_dir = fullfile(corpusdir, current_speech_style, current_gender + "_" + current_speech_style, 'aligned');
                sentence_textgrid_path = fullfile(sentence_textgrid_dir, sentence_textgrid_name);

                sentence_audio_name = sprintf('%s_cleaned_scaled.wav', sentence_id);
                sentence_audio_dir = fullfile(sent_audiodir, current_speech_style, current_gender + "_" + current_speech_style, 'scaled');
                sentence_audio_path = fullfile(sentence_audio_dir, sentence_audio_name);
                
                % Read sentence audio to estimate actual sentence duration.
                [sentence_audio_data, sentence_audio_fs] = audioread(sentence_audio_path);
                sentence_audio_duration = length(sentence_audio_data) / sentence_audio_fs;
                
                % Read phoneme onset times and labels from the TextGrid
                phoneme_info = readPhonesFromTextGrid(sentence_textgrid_path);
                phoneme_onsets_sentence = phoneme_info.t1;
                phoneme_labels = phoneme_info.labels;
                
                % Use the larger of TextGrid and audio duration as sentence length.
                sentence_duration = max(phoneme_info.tmax, sentence_audio_duration);
                sentence_end_time_epoch = sentence_start_time_epoch + sentence_duration;
                
                % Convert sentence boundaries from epoch time to EEG sample indices.
                sentence_slice_start_idx = round(sentence_start_time_epoch * eeg_fs) - baseline_duration_samples + 1;
                sentence_end_idx = round(sentence_end_time_epoch * eeg_fs) + 1;
                
                % Extend the sentence EEG slice so late phonemes keep enough post-onset data.
                post_onset_buffer_samples = prp_duration_samples;
                sentence_slice_end_idx = min(size(block_eeg, 2), sentence_end_idx + post_onset_buffer_samples);
                
                % Extract the EEG segment covering the sentence plus PRP buffer.
                sentence_eeg_slice = block_eeg(:, sentence_slice_start_idx:sentence_slice_end_idx);

                %% extract PRPs for each phoneme
                for phoneme_idx = 1:length(phoneme_labels)
                    current_phoneme_label = phoneme_labels(phoneme_idx);
                    
                    % Skip phonemes not included in the target phoneme set.
                    if ~ismember(current_phoneme_label, string(phonemes_to_include))
                        continue
                    end
                    
                    % Convert phoneme onset to sentence-slice time for PRP indexing.
                    % note: sentence slice starts baseline_duration before sentence onset
                    phoneme_onset_time_slice = phoneme_onsets_sentence(phoneme_idx) + baseline_duration;

                    % Convert phoneme onset to epoch time for output metadata.
                    phoneme_onset_time_epoch = sentence_start_time_epoch + phoneme_onsets_sentence(phoneme_idx);
                    
                    % Define PRP window start and end indices within the sentence slice.
                    prp_start_idx = round(phoneme_onset_time_slice * eeg_fs) - baseline_duration_samples + 1;
                    prp_end_idx = prp_start_idx + prp_window_samples - 1;

                    num_sentence_samples = size(sentence_eeg_slice, 2);
                    
                    % Count PRPs whose baseline starts before the sentence slice.
                    if prp_start_idx < 1
                        prp_trim_begin = prp_trim_begin + 1;
                        continue
                    end

                    % Count PRPs whose response window runs past the sentence slice.
                    if prp_end_idx > num_sentence_samples
                        prp_trim_end = prp_trim_end + 1;
                        continue
                    end
                    
                    % Count invalid windows with reversed or empty bounds.
                    if prp_end_idx <= prp_start_idx
                        prp_invalid_window = prp_invalid_window + 1;
                        continue
                    end
                    
                    % Extract the PRP EEG segment for this phoneme.
                    prp_slice_eeg = sentence_eeg_slice(:, prp_start_idx:prp_end_idx);
                    
                    % Skip PRPs whose extracted length does not match the target window.
                    if size(prp_slice_eeg, 2) ~= prp_window_samples
                        prp_unmatched_duration = prp_unmatched_duration + 1;
                        continue
                    end
                    
                    % Append the PRP EEG trial to the output array.
                    if isempty(prp_trials_eeg)
                        prp_trials_eeg = reshape(prp_slice_eeg, size(prp_slice_eeg,1), size(prp_slice_eeg,2), 1);
                    else
                        prp_trials_eeg(:, :, end+1) = prp_slice_eeg;
                    end
                    
                    % Store phoneme label, onset time, and stimulus metadata for this PRP.
                    prp_phoneme_labels{end+1,1} = current_phoneme_label;
                    prp_onset_times_epoch(end+1,1) = phoneme_onset_time_epoch;
                    prp_emotions{end+1,1} = current_emotion;
                    prp_speech_styles{end+1,1} = current_speech_style;
                    prp_genders{end+1,1} = current_gender;
                end
                
                % Advance epoch time to the start of the next sentence.
                sentence_start_time_epoch = sentence_start_time_epoch + sentence_duration + sentence_isi;
            end

            %% --- Debug: check alignment by comparing cumulative duration with concatenated audio duration ---
            sent_cumulative_dur = sentence_start_time_epoch - pre_event_dur;
            block_audio_scaled_filename = string(block_audio_stem_raw) + string(audio_ext);
            block_audio_scaled_path = fullfile(emo_concatenate_audiodir, block_audio_scaled_filename);
            [block_audio_data, block_audio_fs] = audioread(block_audio_scaled_path);
            block_audio_duration = length(block_audio_data) / block_audio_fs;
            % fprintf('Debug: %s, block %d, duration(sent): %.3fs, duration(concat): %.3fs \n', ...
            %     block_audio_scaled_filename, block_idx, sent_cumulative_dur, block_audio_duration);

        end

        %% summarize PRP extraction
        % Summarize how many PRPs were kept or excluded during extraction.
        total_prps_detected = size(prp_phoneme_labels, 1) + ...
                              prp_trim_begin + ...
                              prp_trim_end + ...
                              prp_invalid_window + ...
                              prp_unmatched_duration;

        fprintf('Total PRPs detected: %d.\n', total_prps_detected);
        fprintf('Skipped PRPs:\n');
        fprintf('  Trimmed at beginning: %d\n', prp_trim_begin);
        fprintf('  Trimmed at end: %d\n', prp_trim_end);
        fprintf('  Invalid window: %d\n', prp_invalid_window);
        fprintf('  Unmatched duration: %d\n', prp_unmatched_duration);
        fprintf('Remaining PRPs: %d.\n', size(prp_phoneme_labels, 1));

        %% optional post-processing
        % Optionally remove linear or constant trends from each PRP trial.
        if detrend_mode == 1 || detrend_mode == 2
            for prp_idx = 1:size(prp_trials_eeg, 3)
                trial_eeg = prp_trials_eeg(:, :, prp_idx);   % [channels x time]
        
                if detrend_mode == 1
                    prp_trials_eeg(:, :, prp_idx) = detrend(trial_eeg')';
                elseif detrend_mode == 2
                    prp_trials_eeg(:, :, prp_idx) = detrend(trial_eeg', 'constant')';
                end
            end
        
            if detrend_mode == 1
                fprintf('Finished linear detrending on PRP trials.\n');
            elseif detrend_mode == 2
                fprintf('Finished constant detrending on PRP trials.\n');
            end
        end
        
        % Optionally subtract the pre-onset baseline from each PRP trial.
        if baseline_correction
            [prp_trials_eeg, ~] = baseline_correct_prp(prp_trials_eeg, baseline_duration_samples);
            fprintf('Finished baseline correction.\n');
        end
        
        % Optionally z-score each PRP trial.
        if standardize
            prp_trials_eeg = standardize_prp(prp_trials_eeg);
            fprintf('Finished z-score standardization on PRP trials.\n');
        end
        
        % Optionally reject noisy PRPs using RMS on selected channels.
        if trial_reject
            [prp_trials_eeg, keep_idx, reject_idx] = RMS_trial_reject_prp( ...
                prp_trials_eeg, reject_threshold, rms_reject_channels, channel_labels);

            prp_phoneme_labels = prp_phoneme_labels(keep_idx);
            prp_onset_times_epoch = prp_onset_times_epoch(keep_idx);
            prp_emotions = prp_emotions(keep_idx);
            prp_speech_styles = prp_speech_styles(keep_idx);
            prp_genders = prp_genders(keep_idx);
        end

        %% prepare output before saving
        % Skip saving if no PRP trials remain after extraction/post-processing.
        if isempty(prp_trials_eeg)
            fprintf('No PRPs extracted for %s. Skipping save.\n', input_file_name);
            continue
        end
        
        % Reorder EEG dimensions to [nPRP x nChan x nTime] for output.
        eegs = permute(prp_trials_eeg, [3 1 2]);   % [nPRP x nChan x nTime]
        
        % Force phoneme labels into an nPRP-by-1 cell array.
        if isstring(prp_phoneme_labels)
            prp_phoneme_labels = cellstr(prp_phoneme_labels(:));
        elseif iscell(prp_phoneme_labels)
            prp_phoneme_labels = prp_phoneme_labels(:);
        else
            prp_phoneme_labels = cellstr(string(prp_phoneme_labels(:)));
        end
        
        % Confirm that EEG trials and phoneme labels have matching counts.
        nPRP = size(eegs, 1);
        if numel(prp_phoneme_labels) ~= nPRP
            error('Mismatch: eegs has %d PRPs but phonemes has %d labels.', nPRP, numel(prp_phoneme_labels));
        end

        %% save output
        % Save PRP EEG, labels, metadata, and extraction statistics to disk.
        output_path = fullfile(subject_output_dir, output_file_name);

        % convert before saving
        prp_phoneme_labels = cellstr(prp_phoneme_labels); % convert "AA" to 'AA'
        prp_emotions = cellstr(prp_emotions);
        prp_speech_styles = cellstr(prp_speech_styles);
        prp_genders = cellstr(prp_genders);

        out = struct();
        out.sub = subject_id;
        out.file = output_file_name;
        out.eegs = eegs;
        out.phonemes = prp_phoneme_labels;
        out.emotions = prp_emotions;
        out.speech_styles = prp_speech_styles;
        out.genders = prp_genders;
        out.label_times = prp_onset_times_epoch;

        out.sr = eeg_fs;
        out.prp_duration = prp_duration;
        out.baseline_duration = baseline_duration;
        out.pre_event_dur = pre_event_dur;
        out.ISI = sentence_isi;
        out.chan_labels = channel_labels(:);
        out.chanlocs = EEG_struct.chanlocs;
        out.prp_trim_begin = prp_trim_begin;
        out.prp_trim_end = prp_trim_end;
        out.prp_invalid_window = prp_invalid_window;
        out.prp_unmatched_duration = prp_unmatched_duration;
        %out.preprocess = preprocess_steps;

        save(output_path, '-struct', 'out', '-v7.3');
        fprintf('Saved: %s | eegs=%s | phonemes=%dx1\n', ...
            output_file_name, mat2str(size(out.eegs)), numel(out.phonemes));
    end
end
