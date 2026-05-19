function save_trigger_csv(outname, trigtimes)

fname = trigtimes{1, 1};
headers = {'filename','trigger','orig_latency_sec','delay_sec','samps_to_add'};
if isempty(trigtimes)
    % No new triggers: skip appending
    fprintf('No new triggers detected for this run. Existing CSV preserved.\n');
    return
end
Tt = cell2table(trigtimes, 'VariableNames', headers);
if exist(outname, 'file')
    Told = readtable(outname);
    % optional: ensure same columns (in case older file differs)
    if ~isequal(Told.Properties.VariableNames, Tt.Properties.VariableNames)
        error('CSV headers do not match expected headers.');
    end
    T = [Told; Tt];
else
    T = Tt;
end

% Write combined table back to CSV
writetable(T, outname);
fprintf('Appended %d new triggers to %s\n', size(Tt,1), outname);
close all;
fprintf('\nDONE. Saved trigger-lag of %s to CSV:\n%s\n', fname, outname);

end