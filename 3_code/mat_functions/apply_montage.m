function EEG = apply_montage(EEG, montname)

if exist(montname, 'file')
    locs = readtable(montname);
    [~, idxEEG, idxLoc] = intersect({EEG.chanlocs.labels}, locs.channel, 'stable'); % preserving order in EEG file, match indices in EEG and montage files
    for a = 1:length(idxEEG)
        EEG.chanlocs(idxEEG(a)).X = locs.x(idxLoc(a));
        EEG.chanlocs(idxEEG(a)).Y = locs.y(idxLoc(a));
        EEG.chanlocs(idxEEG(a)).Z = locs.z(idxLoc(a));
    end
    EEG = eeg_checkset(EEG);
    fprintf('Assigned coordinates for %d channels from %s\n', length(idxEEG), montname);
else
    warning('Montage file not found: %s\n', montname);
end

end