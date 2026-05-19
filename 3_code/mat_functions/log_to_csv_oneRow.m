function log_to_csv_oneRow(csvfile, varargin)
% LOG_TO_CSV_ONEROW
% Append or update a single row in a CSV using name-value pairs.
% Required: 'File', fname
% If fname exists in column 'File', overwrite that row's provided fields.

%% Check inputs
if mod(length(varargin),2) ~= 0
    error('log_to_csv_oneRow requires name-value pairs.');
end

names  = varargin(1:2:end);
values = varargin(2:2:end);

% --- Require 'File' key ---
file_idx = find(strcmpi(names, 'File'), 1);
if isempty(file_idx)
    error('You must provide the pair: ''File'', fname');
end
fname = values{file_idx};

% Convert fname to char string (for matching)
if isnumeric(fname) || islogical(fname)
    fname = num2str(fname);
elseif isempty(fname)
    error('''File'' value (fname) cannot be empty.');
end

% --- Convert inputs to strings (including File) ---
for i = 1:length(values)
    v = values{i};
    if isnumeric(v) || islogical(v)
        values{i} = num2str(v);
    elseif isempty(v)
        values{i} = 'none';
    else
        values{i} = char(v);
    end
end

%% ---- CASE 1: File does not exist or empty ----
if ~exist(csvfile,'file') || dir(csvfile).bytes == 0
    % Ensure header includes File (if user passed weird casing)
    fid = fopen(csvfile,'w');
    fprintf(fid,'%s\n', strjoin(names,','));
    fprintf(fid,'%s\n', format_row(values));
    fclose(fid);
    return
end

%% ---- CASE 2: File exists ----
T = readcell(csvfile);

if isempty(T)
    % treat as empty file
    fid = fopen(csvfile,'w');
    fprintf(fid,'%s\n', strjoin(names,','));
    fprintf(fid,'%s\n', format_row(values));
    fclose(fid);
    return
end

old_header = T(1,:);
old_data   = T(2:end,:);

% Make sure header entries are chars for comparisons
old_header = cellfun(@toCharSafe, old_header, 'UniformOutput', false);

new_header = old_header;

% Ensure there is a 'File' column in existing header (add if missing)
file_col = find(strcmpi(new_header, 'File'), 1);
if isempty(file_col)
    new_header = [{'File'}, new_header];
    file_col = numel(new_header);
    if ~isempty(old_data)
        old_data(:,end+1) = {'none'};
    end
end

% Add missing columns from this call into header
for i = 1:numel(names)
    if ~any(strcmpi(names{i}, new_header))
        new_header{end+1} = names{i};
        if ~isempty(old_data)
            old_data(:,end+1) = {'none'};
        end
    end
end

% If old_data exists but had fewer cols than new_header, pad it
if ~isempty(old_data) && size(old_data,2) < numel(new_header)
    old_data(:, end+1:numel(new_header)) = {'none'};
end

% ---- Find existing row where File == fname ----
row_to_update = [];
if ~isempty(old_data)
    file_col_data = old_data(:, file_col);
    file_col_data = cellfun(@toCharSafe, file_col_data, 'UniformOutput', false);

    row_to_update = find(strcmp(file_col_data, char(fname)), 1); % first match
end

% ---- If row exists: overwrite only provided fields ----
if ~isempty(row_to_update)
    for i = 1:numel(names)
        col = find(strcmpi(new_header, names{i}), 1);
        if ~isempty(col)
            old_data{row_to_update, col} = values{i}; % overwrite
        end
    end

else
    % ---- Else: create a new row aligned to new_header ----
    new_row = repmat({'none'}, 1, numel(new_header));

    for i = 1:numel(names)
        col = find(strcmpi(new_header, names{i}), 1);
        if ~isempty(col)
            new_row{col} = values{i};
        end
    end

    % Append new row
    if isempty(old_data)
        old_data = new_row;
    else
        old_data(end+1, :) = new_row;
    end
end

% ---- Rewrite file (header + all rows) ----
fid = fopen(csvfile,'w');
fprintf(fid,'%s\n', strjoin(new_header,','));

for r = 1:size(old_data,1)
    fprintf(fid,'%s\n', format_row(old_data(r,:)));
end

fclose(fid);

end

%% ------------ helpers ------------
function s = format_row(cells)
cells = cellfun(@toCharSafe, cells, 'UniformOutput', false);
cells = cellfun(@(x)['"' escapeQuotes(x) '"'], cells, 'UniformOutput', false);
s = strjoin(cells,',');
end

function out = toCharSafe(x)
if isempty(x)
    out = 'none';
elseif isnumeric(x) || islogical(x)
    out = num2str(x);
elseif isstring(x)
    out = char(x);
else
    out = char(x);
end
end

function y = escapeQuotes(x)
% CSV-safe: escape embedded double quotes by doubling them
y = strrep(x, '"', '""');
end

