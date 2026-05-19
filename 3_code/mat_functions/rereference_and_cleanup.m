function [EEG, ref_str] = rereference_and_cleanup(EEG, ref_ch)

if ~isempty(ref_ch)
    EEG = pop_reref(EEG, ref_ch); % Re-reference the EEG data
    ref_str = strjoin(ref_ch, ', ');
else
    EEG = pop_reref(EEG, []);
    ref_str = 'average';
end
fprintf('Re-referenced EEG data to: %s\n', ref_str);

% Remove reference channels from dataset if they are present
allowed_refs = {'M1','M2','EXG5'};
refs_to_remove = intersect(ref_ch, allowed_refs);
if ~isempty(refs_to_remove)
    fprintf('Removing reference channels: %s\n', strjoin(refs_to_remove, ', '));
    EEG = pop_select(EEG, 'nochannel', refs_to_remove);
end
EEG.chanlocs = EEG.chanlocs(1:EEG.nbchan);
EEG = eeg_checkset(EEG);

end