% 1. Get the absolute path of the folder containing THIS startup script (3_code)
codeFolder = fileparts(mfilename('fullpath'));

% 2. Construct and add the path for the local 'mat_function' subfolder
matFuncPath = fullfile(codeFolder, 'mat_functions');
if exist(matFuncPath, 'dir')
    addpath(genpath(matFuncPath));
    fprintf('Added local functions path: %s\n', matFuncPath);
else
    warning('Subfolder "mat_function" not found inside: %s', codeFolder);
end

% 3. Go up two levels to find the project root parent folder for EEGLAB
% Level 1 up: ECI1 root folder
projectFolder = fileparts(codeFolder); 
% Level 2 up: The parent directory containing ECI1 and eeglab*
parentFolder = fileparts(projectFolder); 

% 4. Dynamically find and add the EEGLAB folder
searchPattern = fullfile(parentFolder, 'eeglab*');
dirResults = dir(searchPattern);
dirResults = dirResults([dirResults.isdir]);

if ~isempty(dirResults)
    chosenEEGLAB = dirResults(1).name;
    fullEEGLABPath = fullfile(parentFolder, chosenEEGLAB);
    addpath(fullEEGLABPath);
    fprintf('Added EEGLAB path: %s\n', fullEEGLABPath);
    % activate eeglab
    eeglab('nogui'); % Activate EEGLAB
else
    warning('No folder matching "eeglab*" found in: %s', parentFolder);
end
