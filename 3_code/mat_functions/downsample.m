function EEG = downsample(EEG, newsr)

EEG = pop_resample(EEG, newsr);
EEG = eeg_checkset(EEG);
fprintf('Downsampled to %.2f Hz\n', newsr);

end