function tf = should_include_task(task, stim_id)
    if strcmpi(stim_id, 'continuous')
        include_tasks = {'alice','mix','einstein'};
        tf = any(strcmpi(task, include_tasks));
    else
        tf = strcmpi(task, stim_id);
    end
end