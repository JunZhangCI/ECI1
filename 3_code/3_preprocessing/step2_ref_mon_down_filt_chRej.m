clearvars; clc; close all

%% Configurations
group = 'pilot'; % CI or NH or pilot
subs = {'sub-pilot_1'}; % subs that need to be processed e.g., 'sub-pilot_1'
newsr = 128; % downsample rate will be used in following steps
% rawsr = 16384; % speed 7
ref_ch = {'EXG5'}; % mastoid: M1 & M2; Average: {}
disact_ch_CI = {}; % For CI participant, state the disactived EEG channels due to contact with CI here
hp = 1;
lp = 30;
flat_thresh = 3e-6;
% determine the unused channels dynamically
excl_group = {'EXG5', 'EXG6','EXG7','EXG8', 'M1', 'M2', 'Erg1', 'HEOG','VE0G'};
unused_ch = setdiff(excl_group, ref_ch);

%% Path setup
codefolder = fileparts(fileparts(mfilename('fullpath')));
projectfolder = fileparts(codefolder);
if strcmpi(group, 'NH')
    rawdir = fullfile(projectfolder, '2_data', '1_raw', 'NH');
    outdir = fullfile(projectfolder, '2_data', '2_processed', 'NH');
elseif strcmpi(group, 'CI')
    rawdir = fullfile(projectfolder, '2_data', '1_raw', 'CI');
    outdir = fullfile(projectfolder, '2_data', '2_processed', 'CI');
elseif strcmpi(group, 'pilot')
    rawdir = fullfile(projectfolder, '2_data', '1_raw', 'pilot');
    outdir = fullfile(projectfolder, '2_data', '2_processed', 'pilot');
else
    error('%s is not a group of the ECI1 study', group)
end
montagefolder = fullfile(projectfolder, '2_data', '1_raw', 'montage');

% find corresponding montage file
if isempty(ref_ch)
    ref_str = 'avg';
    montdir = select_montage(montagefolder, 'default');
elseif isequal(ref_ch, {'EXG5'})
    ref_str = strjoin(ref_ch, ''); 
    montdir = select_montage(montagefolder, 'nose');
else
    ref_str = strjoin(ref_ch, '');  
    montdir = select_montage(montagefolder, 'default');
end

% temporary stimuli dir for pilot data
stimdir = fullfile(projectfolder, '1_studysetup', 'pilot', 'stimuli');

%% Sub selection
% if no subjects defined, automatically detect folders
if isempty(subs)
    subfolders=dir(fullfile(rawdir,'sub*'));
    subfolders=subfolders([subfolders.isdir]); % keep only directories
    subs = {subfolders.name}; % keep subject names as strings
end
fprintf('\nSubjects to process:\n');
disp(subs);

%% Loop over subjects
for sub = 1:length(subs)
    subname = subs{sub};
    sub_rawdir = fullfile(rawdir, subname);
    if ~exist(sub_rawdir, 'dir')
        error('Raw directory does not exist: %s', sub_rawdir);
    end
    sub_log_dir = fullfile(outdir, subname);
    sub_outdir = fullfile(outdir, subname, 'ref_down_filt_chRej'); % Output directory for the subject
    if ~exist(sub_outdir, 'dir')
        mkdir(sub_outdir)
    end
    sub_raw_files = find_subject_files(sub_rawdir, 'bdf', 'emo');
    logfile = fullfile(sub_log_dir, [subname '_preprocess_log.csv']);
    if ~exist(logfile,'file')
        fid = fopen(logfile,'w');
        fclose(fid);
    end
    sub_lag_dir = fullfile(outdir, subname);

%% Loop over files
    for file = 1:length(sub_raw_files)
        bdf_file_path = sub_raw_files{file};
        [~, fname] = fileparts(bdf_file_path);
        % find corresponding order excel
        namepts = strsplit(fname,'_');
        acqfull = namepts{5};
        acqnum = regexp(acqfull,'\d+','match','once');
        order = str2double(acqnum);
        order_excel = fullfile(stimdir, 'orders', sprintf('order%d.xlsx', order));
        % create out path
        outname = sprintf('%s_down%dHz_ref%s_filt%d-%dHz_chRej.set', fname, newsr, ...
            ref_str, hp, lp);
        if exist(fullfile(sub_outdir, outname), 'file')
            fprintf('Skipping %s — already processed.\n', outname);
            continue
        end
        fprintf('\n--- Processing %s ---\n', fname);

        % Process the file
        EEG = load_and_montage(bdf_file_path, montdir);
        EEG.preprocess = {};
        EEG.preprocess{end+1} = 'montage';
        EEG = assign_ch_type(EEG);
        EEG.preprocess{end+1} = 'assign_ch_type';
        EEG = remove_ch(EEG, unused_ch, 0);
        EEG.preprocess{end+1} = 'remove_unused_ch';
        [EEG, ref_str] = rereference_and_cleanup(EEG, ref_ch);       
        EEG.preprocess{end+1} = 'rereference';
        EEG = downsample(EEG, newsr);
        EEG.preprocess{end+1} = 'downsample';
        EEG = pop_eegfiltnew(EEG, hp, []); 
        EEG = pop_eegfiltnew(EEG, [], lp); 
        EEG.preprocess{end+1} = 'filter';
        [EEG, removed_str, flat_str] = manual_chRej(EEG, fname, flat_thresh);        
        EEG.preprocess{end+1} = 'manual_ch_rej';        
        EEG = update_triggers_from_csv(sub_lag_dir, fname, EEG);
        EEG.preprocess{end+1} = 'update_trigger';
        if ~isempty(disact_ch_CI)
            EEG = remove_ch(EEG, disact_ch_CI, 1);
            EEG.preprocess{end+1} = 'remove_CI_ch';
        end
        EEG = trim_eeg(EEG, stimdir, order_excel);
        EEG.preprocess{end+1} = 'trim_gaps';

        % save the processed file
        disact_str = strjoin(disact_ch_CI, ' ');  % for logging       
        save_processed_eeg(EEG, outname, sub_outdir);
        date_str = string(datetime('now','Format','yyyy-MM-dd'));
        time_str = string(datetime('now','Format','HH:mm'));
        
        log_to_csv_oneRow( logfile,...
            'File', fname, ...
            'Date', date_str, ...
            'Time', time_str, ...
            'Ref_ch', ref_str, ...
            'Sample_rate', newsr, ...
            'low_cutoff', hp, ...
            'high_cutoff', lp, ...
            'Disact_ch', disact_str, ...
            'Flat_ch', flat_str,...
            'Removed_ch', removed_str);
        close all
    end
end