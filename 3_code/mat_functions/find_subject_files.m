function files = find_subject_files(folder_dir, file_ext, stim_id)

% make stim_id optional
if nargin < 2 || isempty(stim_id)
    stim_id = [];
end
% Ensure extension does not include dot
file_ext = strrep(file_ext, '.', '');
% Find matching files
s = dir(fullfile(folder_dir, ['*.' file_ext]));
% Check corresponding task
files = {};
for i = 1:length(s)
    fname = s(i).name;
    task  = parse_task_from_filename(fname);
    if isempty(stim_id) || should_include_task(task, stim_id)
        files{end+1} = fullfile(folder_dir, fname);
    end
end
if isempty(files) & ~isempty(stim_id)
    fprintf('no task-%s file with ".%s" extension is found in folder: \n %s\n', ...
        stim_id, file_ext, folder_dir)
elseif isempty(files) & isempty(stim_id)
    fprintf('No files with ".%s" extension found in folder:\n%s\n', ...
        file_ext, folder_dir)
elseif ~isempty(files) & ~isempty(stim_id)
    fprintf('Found %d task-%s files with ".%s" extension in folder: \n %s\n', ...
        length(files), stim_id, file_ext, folder_dir);
elseif ~isempty(files) & isempty(stim_id)
    fprintf('Found %d files with ".%s" extension in folder:\n%s\n', ...
        length(files), file_ext, folder_dir);
end

end