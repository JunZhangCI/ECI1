function save_processed_eeg(EEG, outname, out_dir)

EEG.data = double(EEG.data);
EEG = pop_saveset(EEG, ...
        'filename', outname, ...
        'filepath', out_dir, ...
        'savemode', 'onefile');
fprintf('Saved processed file: %s\n', fullfile(out_dir, outname));

end