function EEG = update_triggers_from_csv(csv_dir, EEG_fname, EEG, exclude_event)
% UPDATE_TRIGGERS_FROM_CSV(maindir, EEG_fname, EEG, exclude_event)
% Updates event triggers in an EEGLAB EEG dataset based on corrected trigger
% latencies stored in a CSV file for the corresponding subject and task.

%% Validate parameters and set defaults
if isempty(EEG)
    error('An EEGLAB EEG struct should be entered');
end
if ~exist('exclude_event','var') || isempty(exclude_event)
    exclude_event = [];
end

fprintf('Processing file: %s\n', EEG_fname);
namepts = strsplit(EEG_fname, '_');

%% Locate and Filter CSV Files
csv_files = dir(fullfile(csv_dir, '*.csv'));
if isempty(csv_files)
    error('No CSV files found in %s', csv_dir);
end

% Determine task directly
task = erase(namepts{4}, 'task-');

% Vectorized search for matching task files instead of a manual for-loop
csv_names = {csv_files.name};
valid_csv = csv_names(contains(csv_names, ['_' task], 'IgnoreCase', true));

% Resolve file selection
if isempty(valid_csv)
    error('No matching CSV for task: %s', task);
elseif length(valid_csv) == 1
    selected_file = valid_csv{1};
else
    fprintf('Multiple CSV files found:\n');
    for f = 1:length(valid_csv)
        fprintf('%d: %s\n', f, valid_csv{f});
    end
    choice = input(sprintf('Select a file (1-%d): ', length(valid_csv)));
    if choice < 1 || choice > length(valid_csv)
        error('Invalid selection. Please run again and choose a valid index.');
    end
    selected_file = valid_csv{choice};
end

trigtimes_file = fullfile(csv_dir, selected_file);
trigtimes = readtable(trigtimes_file);
fprintf('CSV file selected: %s\n', trigtimes_file);

EEG = eeg_checkset(EEG);
fprintf('Loaded %s: %d channels, %.1f Hz\n', EEG_fname, EEG.nbchan, EEG.srate);

%% Optimized Event Cleaning
evt_raw = {EEG.event.type};

% Completely vectorized method to unwrap nested cells and convert types
if iscell(evt_raw{1})
    evt_raw = cellfun(@(x) x{1}, evt_raw, 'UniformOutput', false);
end

if isnumeric(evt_raw{1})
    evt_nums = cell2mat(evt_raw);
else
    % Strip strings and convert directly in array format
    clean_strs = regexprep(evt_raw, '[^\d.-]', '');
    evt_nums = str2double(clean_strs);
end

%% Interactive Graphical Exclusion Setup
if isempty(exclude_event) && length(unique(evt_nums)) > 1
    unique_evts = unique(evt_nums(~isnan(evt_nums)));
    event_str = strjoin(cellstr(num2str(unique_evts(:))), ', '); % Uses clean strjoin (no trailing commas)
    
    msg_text = sprintf('Current Events: %s\n\nDo you want to exclude any triggers?', event_str);
    ok = questdlg(msg_text, 'Exclude Triggers', 'Yes', 'No', 'No');
    
    if strcmpi(ok, 'Yes')
        valid_input = false;
        while ~valid_input
            prompt = {'Enter event codes to exclude (e.g., 1 2 or 1,2):'};
            user_cell = inputdlg(prompt, 'Specify Exclusions', [1 50], {''});
            
            if isempty(user_cell), disp('Exclusion canceled.'); break; end
            
            exclude_event = str2num(user_cell{1}); %#ok<ST2NM>
            if ~isempty(exclude_event) || isempty(strtrim(user_cell{1}))
                valid_input = true;
            else
                warndlg('Invalid format! Use numbers separated by spaces or commas.', 'Input Error');
            end
        end
    end
end

%% Single-Pass Filtering Rules
valid_idx = ~isnan(evt_nums);
if ~isempty(exclude_event)
    excluded_idx = find(ismember(evt_nums, exclude_event));
    fprintf('Excluded event indices in EEG.event: [%s]\n', num2str(excluded_idx));
    valid_idx = valid_idx & ~ismember(evt_nums, exclude_event);
end

evt_nums = evt_nums(valid_idx);
EEG.event = EEG.event(valid_idx);

if isempty(EEG.event)
    error('No valid events found in %s after filtering.', EEG_fname);
end

% Vectorized cell assignment instead of standard loop syntax
str_nums = arrayfun(@num2str, evt_nums, 'UniformOutput', false);
[EEG.event.type] = str_nums{:};

fprintf('Events: %s\n', num2str(unique(evt_nums)));

%% Update Trigger Latencies
rawname = extractBefore(EEG_fname, '_down');
if isempty(rawname), rawname = strjoin(namepts(1:5), '_'); end

Tsub = trigtimes(contains(trigtimes.filename, rawname), :);

for ev = 1:length(EEG.event)
    this_trig = evt_nums(ev);
    row_idx = find(Tsub.trigger == this_trig, 1, 'first');
    
    if isempty(row_idx)
        warning('No CSV row found for trigger %d (event %d). Skipping.', this_trig, ev);
        continue;
    end
    
    EEG.event(ev).latency = EEG.event(ev).latency + Tsub.samps_to_add(row_idx);
    Tsub(row_idx, :) = []; % Pop row to avoid duplicate match overlaps
end

EEG = eeg_checkset(EEG);
fprintf('Updated %d trigger latencies for %s.\n', length(EEG.event), EEG_fname);
end
