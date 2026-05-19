function [Xbc, baseline_vals] = baseline_correct_prp(X, nBaseSamp)
% BASELINE_CORRECT_PRP Baseline-correct PRP windows.
%
%   [Xbc, baseline_vals] = baseline_correct_prp(X, nBaseSamp)
%
%   Inputs
%     X         : [nChan x nTime x nPRP] double
%     nBaseSamp : number of baseline samples (e.g., baseline_duration_samp)
%
%   Outputs
%     Xbc          : baseline-corrected data, same size as X
%     baseline_vals: [nChan x 1 x nPRP] baseline mean that was subtracted

X = double(X);

if ndims(X) ~= 3
    error('X must be 3-D: [nChan x nTime x nPRP].');
end

[~, nTime, ~] = size(X);

if nBaseSamp < 1 || nBaseSamp > nTime
    error('nBaseSamp must be between 1 and nTime. Got %d, nTime=%d.', nBaseSamp, nTime);
end

base_seg = X(:, 1:nBaseSamp, :);                         % [nChan x nBaseSamp x nPRP]
baseline_vals = mean(base_seg, 2, 'omitnan');            % [nChan x 1 x nPRP]

Xbc = X - baseline_vals;                                 % implicit expansion (R2016b+)
end