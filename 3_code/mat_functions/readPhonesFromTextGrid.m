function info = readPhonesFromTextGrid(textgrid_file)

% format check on arguement input
arguments
    textgrid_file (1,1) string
end

% check existence of target file
if ~isfile(textgrid_file)
    error('File does not exist: %s', textgrid_file);
end

% read the content of the TextGrid file
txt = fileread(textgrid_file);

% get the 'phones' tier
tier_pattern = 'name\s*=\s*"phones"'; % locate phase like 'name = "phones"'(\s*: any number of spaces)
tier_pos = regexp(txt, tier_pattern, 'once'); 
if isempty(tier_pos)
    error('No "phones" tier found in the TextGrid file: %s', textgrid_file);
end

% extract the relevant section for the 'phones' tier
phones_section = txt(tier_pos:end);

% identify interval of each phoneme
% locate structure:
%   xmin = 0.01
%   xmax = 0.5
%   text = "HH"
% Tip: only content in () will be saved as tokens by regexp
%   - ([0-9eE\+\-\.]+): 0-9 --> accept digits
%                       eE --> accept scientific notations
%                       \+\- --> accept minus/plus sign
%                       \. --> accept decimal point
%                       + --> accept multiple characters
%   - (.*?): .* --> any single character that repeat one or more times
%            ?  --> have the matching stop at next " mark
interval_pattern = ['xmin\s*=\s*([0-9eE\+\-\.]+)\s*[\r\n]+' ...
                    '\s*xmax\s*=\s*([0-9eE\+\-\.]+)\s*[\r\n]+' ...
                    '\s*text\s*=\s*"([A-Za-z]*)\d*"']; % text = will accept ""

% Extract intervals and phoneme texts using regexp
tokens = regexp(phones_section, interval_pattern, 'tokens');

% Check if intervals were found
if isempty(tokens)
    error('No intervals found in the "phones" tier of the TextGrid file: %s', textgrid_file);
end

% initialize output structure
n = numel(tokens);
t1 = zeros(n,1);
t2 = zeros(n,1);
labels = strings(n,1);

% rearrange t1, t2 values into arrays
for i = 1:n
    t1(i) = str2double(tokens{i}{1});
    t2(i) = str2double(tokens{i}{2});
    labels(i) = string(tokens{i}{3});
end
labels = strip(labels);

% extract sentence ID from filename
[~, name, ~] = fileparts(textgrid_file);
namepts = strsplit(name, '_');
sent_ID = str2double(namepts{2});

% Populate the output structure with extracted data
info = struct();
info.sentence_ID = sent_ID;
info.tmin = t1(1);
info.tmax = t2(end);
info.t1 = t1;
info.t2 = t2;
info.labels = labels;

end
