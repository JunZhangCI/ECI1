function EEG = assign_ch_type(EEG)

for c = 1:length(EEG.chanlocs)
    lbl = EEG.chanlocs(c).labels;
    if strcmpi(lbl,'HEOG') || strcmpi(lbl,'VE0G') 
        EEG.chanlocs(c).type = 'EOG'; % eye channels
    elseif strcmpi(lbl, 'Erg1')
        EEG.chanlocs(c).type = 'STIM'; % stimulus channels
    elseif strcmpi(lbl,'M1') || strcmpi(lbl,'M2') || strcmpi(lbl,'EXG5')
        EEG.chanlocs(c).type = 'REF'; % reference channels
    else
        EEG.chanlocs(c).type = 'EEG'; % EEG channels
    end
end

end