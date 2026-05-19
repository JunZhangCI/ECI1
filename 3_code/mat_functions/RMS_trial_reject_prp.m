function [X_clean, keep_idx, reject_idx, rms_vals, threshold] = RMS_trial_reject_prp(X, reject_thre, prp_ch, chan_labels)
% RMS_TRIAL_REJECT_PRP Reject PRPs using RMS computed from selected channels
%
%   [X_clean, keep_idx, reject_idx, rms_vals, threshold] = ...
%       RMS_trial_reject_prp(X, reject_thre, prp_ch, chan_labels)
%
% Inputs
%   X           : [nChan x nTime x nPRP] PRP EEG
%   reject_thre : threshold in SD units (e.g., 2)
%   prp_ch      : channels to use for RMS
%                - numeric indices (e.g., [1 3 5]), OR
%                - labels (cellstr/string) if chan_labels provided
%   chan_labels : (optional) channel labels, length nChan (cellstr/string)
%
% Outputs
%   X_clean     : PRPs after rejection (all channels kept, only bad epochs removed)
%   keep_idx    : indices of kept PRPs
%   reject_idx  : indices of rejected PRPs
%   rms_vals    : 1 x nPRP RMS value per PRP (based on selected channels)
%   threshold   : scalar RMS threshold used

if nargin < 4
    chan_labels = [];
end

X = double(X);
[nChan, nTime, nPRP] = size(X);

% resolve prp_ch to numeric indices
if isempty(prp_ch)
    ch_idx = 1:nChan;  % default: all channels
elseif isnumeric(prp_ch)
    ch_idx = prp_ch(:)';
else
    % labels provided in prp_ch, need chan_labels
    if isempty(chan_labels)
        error('If prp_ch is labels, you must provide chan_labels.');
    end
    ch_idx = find(ismember(string(chan_labels), string(prp_ch)));
end

% validate indices
ch_idx = unique(ch_idx);
ch_idx = ch_idx(ch_idx >= 1 & ch_idx <= nChan);
if isempty(ch_idx)
    error('No valid channels found for RMS rejection.');
end

% compute RMS per PRP using only selected channels
Xsel = X(ch_idx, :, :);                        % [nSel x nTime x nPRP]
X2 = reshape(Xsel, numel(ch_idx)*nTime, nPRP); % [nSel*nTime x nPRP]
rms_vals = sqrt(mean(X2.^2, 1));               % [1 x nPRP]

% compute threshold and reject
rms_mean = mean(rms_vals);
rms_std  = std(rms_vals);
threshold = rms_mean + reject_thre * rms_std;

reject_idx = find(rms_vals > threshold);      % Indices of rejected PRPs
keep_idx = setdiff(1:nPRP, reject_idx);       % Indices of kept PRPs
X_clean = X(:, :, keep_idx);                   % Cleaned PRPs with bad epochs removed

fprintf('RMS rejection (channels used=%d): %d/%d removed (%.2f%%). Threshold=%.4g\n', ...
    numel(ch_idx), numel(reject_idx), nPRP, 100*numel(reject_idx)/nPRP, threshold);

end