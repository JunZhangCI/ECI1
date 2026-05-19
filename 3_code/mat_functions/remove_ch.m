function EEG = remove_ch(EEG, remove_chs, do_interp)

% Set default of do_interp to be 0 (off)
if ~exist('do_interp','var') || isempty(do_interp)
    do_interp = 0;
end

if isempty(remove_chs)
    return
end

% find which requested channels exist in EEG
existing_chs = intersect(remove_chs, {EEG.chanlocs.labels});
if isempty(existing_chs)
    fprintf('No matching channels found for removal.\n');
    return
end

% Keep original layout of EEG-only channels for interpolation
if do_interp
    orig_chanlocs = EEG.chanlocs;
    orig_types    = {EEG.chanlocs.type};
end

%remove channels
EEG = pop_select(EEG,'nochannel', existing_chs);
EEG.chanlocs = EEG.chanlocs(1:EEG.nbchan);
EEG = eeg_checkset(EEG);
fprintf('Removed channels(%d): %s\n', length(existing_chs), strjoin(existing_chs, ', '));

% Optionally interpolate removed EEG channels only
if do_interp
    remove_EEG_chs = ismember(existing_chs, {orig_chanlocs(strcmpi(orig_types, 'EEG')).labels});
    if any(remove_EEG_chs)
        EEG = pop_interp(EEG, orig_chanlocs, 'spherical');
        EEG = eeg_checkset(EEG);
        fprintf('Interpolated removed EEG channels back in.\n');
        fprintf('Channels remaining (%d): %s\n', ...
        EEG.nbchan, strjoin({EEG.chanlocs.labels}, ', '));
    else
        fprintf('No removed EEG channels to interpolate (skipping interp).\n');
    end
end

end