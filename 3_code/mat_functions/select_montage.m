function montname = select_montage(montage_folder, mont_type)

valid_mont = {'nose','default'};
assert(any(strcmpi(mont_type, valid_mont)),'mont_type must be one of: nose, default');

switch lower(mont_type)
    case 'nose'
        % Handle the 'nose' montage type
        montname = fullfile(montage_folder, 'biosemi_32ch_2mastoid_EXG5nose_locs_EEGLAB.csv'); % Montage file for nose type
    case 'default'
        montname = fullfile(montage_folder, 'biosemi_32ch_2mastoid_locs_EEGLAB.csv'); % Montage file for mastoid type
end
if ~exist(montname,'file')
    error('Montage file not found: %s', montname)
end
% Return the montage name if the file exists
end