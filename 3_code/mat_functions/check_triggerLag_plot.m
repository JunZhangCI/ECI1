function debugPlotsFlag = check_triggerLag_plot( ...
        signal, tempstim, ...
        delay_EEG_samples, win_onset_offset_samp, ...
        tri)

% SHOW_DEBUG_ALIGNMENT_PLOT
% Ask user whether to display debug alignment plot.
% Returns true if user selected "No".

if ~exist('win_onset_offset','var') || isempty(win_onset_offset_samp)
    win_onset_offset_samp = 0;
end

choice = questdlg('Show debug plots for this peak?', ...
                  'Debug Plot Option', ...
                  'Yes','No','No');

if strcmp(choice,'Yes')

    % Normalize
    sig_norm  = signal   / max(abs(signal));
    temp_norm = tempstim / max(abs(tempstim));

    % Shift WAV by delay_EEG_samples
    aligned_signal = zeros(size(temp_norm));
    start_idx = 1 + delay_EEG_samples - win_onset_offset_samp;
    end_idx   = start_idx + length(sig_norm) - 1;

    if start_idx > 0 && end_idx <= length(aligned_signal)
        aligned_signal(start_idx:end_idx) = sig_norm;
    end

    % Plot
    figure;
    hold on;
    plot(temp_norm, 'k', 'DisplayName','EEG Stim Segment');
    plot(aligned_signal,'r','LineWidth',1.2,'DisplayName','WAV Stim (shifted)');
    title(sprintf('Peak %d — Corrected Alignment', tri),'Interpreter','none');
    xlabel('Samples');
    ylabel('Normalized Amplitude');
    legend;
    grid on;
    pause;

    debugPlotsFlag = true;

else
    debugPlotsFlag = false;
end

end