function [EEG_cleaned, rejectedStr, IC_shown] = run_ica_manual_reject( ...
    EEG, focus_ch, ica_mode, IC_shown_default)

if ~exist('IC_shown_default','var') || isempty(IC_shown_default)
    IC_shown_default = 32;
end

%% Select EEG channels only
eeg_idx = find(strcmpi({EEG.chanlocs.type}, 'EEG'));
EEG = pop_select(EEG, 'channel', eeg_idx);

%% Run selected mode of ICA
EEG_wICA = pop_runica(EEG, 'icatype', ica_mode);

% Save the uncleaned snapshot containing ICA weights for later comparison
EEG_uncleaned = EEG_wICA; 

%% Pre-slice focus channels 
focus_ch_arr = strsplit(strtrim(focus_ch));
existingCh = intersect(focus_ch_arr, {EEG_uncleaned.chanlocs.labels});

if ~isempty(existingCh)
    % Slice "Before" channels once here outside the loop
    EEG_unc = pop_select(EEG_uncleaned, 'channel', existingCh);
else
    warning('None of the focus channels exist in this dataset.');
    EEG_unc = [];
end

%% Choose how many ICs to inspect
IC_shown = ask_n_IC_show(IC_shown_default);

%% Manual rejection loop
done = false;
rejectedStr = 'None';

while ~done
    % Always fetch our base uncleaned data to execute/re-execute rejections cleanly
    EEG_cur = EEG_uncleaned; 
    % Run ICLabel and capture the updated EEG structure with classifications
    EEG_cur = pop_iclabel(EEG_cur, 'default'); 
    assignin('base', 'EEG', EEG_cur);  

    % Open interactive layout grid
    pop_viewprops(EEG_cur, 0, 1:IC_shown, {'freqrange', [1 50]}, {}, 1, 'ICLabel');

    %% User selects ICs
    user_input = get_IC_rejection_gui();
    
    if ~isempty(user_input)
        comps = str2num(user_input); %#ok<ST2NM>
        % Apply the component removal
        EEG_cleaned = pop_subcomp(EEG_cur, comps);
        EEG_cleaned = eeg_checkset(EEG_cleaned);
        rejectedStr = user_input;
    else
        EEG_cleaned = EEG_cur;
        rejectedStr = 'none';
        done = true;
        close all;
        continue; % Exit loop safely if cancelled
    end
    
    %% Compare before/after on focus channels
    if ~isempty(existingCh) && ~isempty(EEG_unc)
        EEG_cln = pop_select(EEG_cleaned, 'channel', existingCh);
        
        % Open localized comparison charts
        pop_eegplot(EEG_unc, 1, 1, 1, [], 'title', 'Before ICA', 'spacing', 100);
        pop_eegplot(EEG_cln, 1, 1, 1, [], 'title', 'After ICA', 'spacing', 100);

        uiwait(msgbox('Review the open scrolling windows, then close this box to proceed.', 'Review Traces'));
    else
        warning('None of the focus channels exist in this dataset.');
    end
    
    %% Optional grand ERP plot (epoched only)
    choice1 = questdlg( ...
        'Do you want to check the grand ERP?', ...
        'Inspect ERP', ...
        'Yes', 'No', 'No');

    if strcmp(choice1, 'Yes') && ndims(EEG_cleaned.data) == 3
        times = EEG_cleaned.times; % ms
        grand_erp_unc = mean(mean(EEG_unc.data, 3), 1);
        grand_erp_cln = mean(mean(EEG_cln.data, 3), 1);

        figure;
        plot(times, grand_erp_unc, 'k', 'LineWidth', 1.2);
        hold on;
        plot(times, grand_erp_cln, 'b', 'LineWidth', 1.2);
        xline(0, '--r', 'LineWidth', 1.5);
        hold off;

        xlabel('Time (ms)');
        ylabel('Amplitude (\muV RMS)');
        legend('Before ICA', 'After ICA');
        title(['Grand ERP, Components removed: ', rejectedStr]);
        grid on;
    end
    pause;
    
    % Acceptance check prompt
    choice2 = questdlg( ...
        'Accept cleaning configuration?', ...
        'End ICA Process', ...
        'Yes', 'No', 'No');
        
    if strcmp(choice2, 'Yes')
        done = true;
    else
        close all; % Wipes the figures so a clean revision can run
    end
end

close all;
end

%% ===================== HELPER FUNCTIONS =====================

function IC_shown = ask_n_IC_show(IC_shown_default)
valid_input = false;
while ~valid_input
    answer = inputdlg( ...
        sprintf('Enter number of components to inspect (default=%d):', IC_shown_default), ...
        'Select IC Count', ...
        [1 60]);
        
    if isempty(answer)
        IC_shown = IC_shown_default;
        break;
    end
    
    input_str = strtrim(answer{1});
    if isempty(input_str)
        IC_shown = IC_shown_default;
        break;
    end
    
    IC_shown = str2double(input_str);
    
    if ~isnan(IC_shown) && IC_shown > 0
        IC_shown = round(IC_shown); % ensure integer
        valid_input = true;
    else
        uiwait(warndlg('Please enter a valid positive number.', 'Invalid Input'));
    end
end
fprintf('Topographs of first %d components will be shown.\n', IC_shown);
end

function user_input = get_IC_rejection_gui()
user_input = '';
d = dialog('Name', 'Select ICs to Reject', ...
    'Position', [300 300 300 120], ...
    'WindowStyle', 'normal'); % NON-MODAL

uicontrol('Parent', d, ...
    'Style', 'text', ...
    'Position', [20 70 260 20], ...
    'String', 'Enter components to remove (e.g., [1 3 5]):');
    
editBox = uicontrol('Parent', d, ...
    'Style', 'edit', ...
    'Position', [20 45 260 25], ...
    'HorizontalAlignment', 'left');
    
uicontrol('Parent', d, ...
    'Position', [50 10 80 25], ...
    'String', 'OK', ...
    'Callback', @(src,evt) uiresume(d));
    
uicontrol('Parent', d, ...
    'Position', [170 10 80 25], ...
    'String', 'Cancel', ...
    'Callback', @(src,evt) delete(d));

uiwait(d); % pause execution but allow figure interaction

if isvalid(editBox)
    user_input = strtrim(editBox.String);
end
if isvalid(d)
    delete(d);
end
end
