function task = parse_task_from_filename(fname)
    % expects something like: sub-xxx_ses-xx_task-da_run-1_... .bdf
    [~, stem, ~] = fileparts(fname);
    parts = strsplit(stem, '_');

    % your original code used parts{4} as task-*; keep that behavior
    if numel(parts) >= 4
        taskfull = parts{4};          % e.g. "task-da"
        task = erase(taskfull,'task-'); % -> "da"
    else
        task = '';
    end
end