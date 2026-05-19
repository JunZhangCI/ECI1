function order = getSentenceOrderFromTable(xlsx_path, target_cont_audio_name)

% format check on arguement input
arguments
    xlsx_path (1,1) string
    target_cont_audio_name (1,1) string
end

% check existence of target file
if ~isfile(xlsx_path)
    error('File does not exist: %s', xlsx_path)
end

% read table from xlsx and locate row of target audio file
order_tab = readtable(xlsx_path);
cont_audio_names = order_tab.output_filename;
order_list = order_tab.file_order;
row_idx = find(cont_audio_names == target_cont_audio_name, 1);
if isempty(row_idx)
    error('Target audio file not found in table: %s', target_cont_audio_name)
end

% extract sentence audio file list from file_order column and trim them
target_order_list = order_list{row_idx};
target_order_list = strtrim(split(target_order_list,';'));
for sent = 1:length(target_order_list)
    current_audio_name = target_order_list(sent);
    namepts = split(current_audio_name,'_');
    core_info = namepts(1:4);
    core_info_cont = strjoin(core_info,'_');
    target_order_list{sent} = core_info_cont; % only keep core info: "54_06_hap_f"
end

% assign the processed audio names to the output variable
order = target_order_list;

end