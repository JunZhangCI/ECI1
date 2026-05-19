clearvars; clc; close all

%% Configurations
group = 'pilot'; % CI or NH or pilot
subs = {'sub-pilot_1'}; % subs that need to be processed e.g., 'sub-pilot_1'
focus_ch = 'AF3 AF4 Fz F3 F4 FC1 FC2';
first_ica_mode = 'sobi'; 
second_ica_mode = 'sobi';

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
    sub_log_dir = fullfile(datadir, subname);
    sub_indir = fullfile(datadir, subname, 'ref_down_filt_chRej');
    sub_outdir = fullfile(sub_indir, 'ica');
    if ~exist(sub_outdir,'dir')
        mkdir(sub_outdir);
    end
    logfile = fullfile(sub_log_dir, [subname '_preprocess_log.csv']);
    if ~exist(logfile,'file')
        fid = fopen(logfile,'w');
        fclose(fid);
    end
    sub_in_files = find_subject_files(sub_indir, 'set', 'emo');

    %% Loop over file
    for file = 1:length(sub_in_files)
        set_file_path = sub_in_files{file};
        [~, fname] = fileparts(set_file_path);
        % Split by underscore
        parts = strsplit(fname, '_');
        % Rejoin the first 5 parts (sub, ses, task, emo, acq-1) with underscores
        raw_fname = strjoin(parts(1:5), '_');
        outname = sprintf('%s_ica.set', fname);
        if exist(fullfile(sub_outdir, outname), 'file')
            fprintf('Skipping %s — already processed.\n', outname);
            continue
        end
        EEG = pop_loadset(set_file_path);
        [EEG_cleaned, rejectedStr, ~] = run_ica_manual_reject( ...
            EEG, focus_ch, first_ica_mode);
        EEG_cleaned.preprocess{end+1} = 'ica';
        save_processed_eeg(EEG_cleaned, outname, sub_outdir)
        log_to_csv_oneRow(logfile, ...
            'File', raw_fname, ...
            'IC_reject1', rejectedStr);
        out_EEG = EEG_cleaned;
        choice = questdlg( ...
        'Do you want to run ICA on this data again?', ...
        'Rerun ICA', ...
        'Yes','No','No');
        if strcmp(choice,'Yes')
            [EEG_recleaned, rejectedStr_reclean, ~] = run_ica_manual_reject( ...
            EEG_cleaned, focus_ch, second_ica_mode);
            [~, base, ~]= fileparts(outname);
            reclean_outname = sprintf('%s_ica2.set', base);
            EEG_recleaned.preprocess{end+1} = 'ica2';
            save_processed_eeg(EEG_recleaned, reclean_outname, sub_outdir);
            log_to_csv_oneRow(logfile, ...
                'File', raw_fname, ...
                'IC_reject2', rejectedStr_reclean);
            out_EEG = EEG_recleaned;
        end
        % save a copy as a mat file
        mat_filename = fullfile(sub_outdir, [raw_fname '.mat']);
        save_pack = struct('EEG', out_EEG);
        save(mat_filename, '-struct', 'save_pack', 'EEG', '-v7.3');
        close all
    end
end
