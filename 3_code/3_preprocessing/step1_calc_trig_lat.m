clearvars; clc;

%% Configuration
group = 'pilot'; % CI or NH or pilot
subs = {'sub-pilot_1'}; % subs that need to be processed e.g., 'sub-pilot_1'
newsr = 128; %Hz

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
% temporary stimuli dir for pilot data
stimdir = fullfile(projectfolder, '1_studysetup', 'pilot', 'stimuli');
%stimdir = fullfile(projectfolder, '1_studysetup','stimuli');

%% Sub selection
% if no subjects defined, automatically detect folders
if isempty(subs)
    subfolders=dir(fullfile(rawdir,'sub*'));
    subfolders=subfolders([subfolders.isdir]); % keep only directories
    subs = {subfolders.name}; % keep subject names as strings
end
fprintf('Processing group: %s\n', group);
fprintf('Subjects to process:\n');
disp(subs);

%% Calculating trigger latency 
for sub = 1:length(subs)
    sub_indir = fullfile(rawdir, subs{sub});
    if ~exist(sub_indir, 'dir')
        error('Raw directory does not exist: %s', sub_indir);
    end
    sub_outdir = fullfile(outdir, subs{sub}); % Output directory for the subject
    if ~exist(sub_outdir, 'dir')
        mkdir(sub_outdir)
    end
    outcsv = fullfile(sub_outdir, sprintf('%s_emo_adjusted_triggertimes_%dHz.csv', ...
        subs{sub}, newsr));
    [~, fdone] = load_existing_csv(outcsv);
    sub_raw_files = find_subject_files(sub_indir, 'bdf', 'emo');
    
    for file = 1:length(sub_raw_files)
        bdf_file_path = sub_raw_files{file};
        [~, fname] = fileparts(bdf_file_path);
        if ismember(fname, fdone)
            fprintf('Skipping already processed file: %s\n', fname);
            continue
        end
        trigtimes = compute_trigger_lag_emo(bdf_file_path, stimdir, newsr, 1);
        save_trigger_csv(outcsv, trigtimes);
        close all
    end
end
