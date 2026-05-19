function EEG = load_and_montage(bdf_file_path, montname)

EEG = pop_biosig(bdf_file_path);
EEG = eeg_checkset(EEG);

if exist(montname,'file')
    EEG = apply_montage(EEG, montname);
end

end