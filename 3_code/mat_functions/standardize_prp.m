function Xz = standardize_prp(X)
% STANDARDIZE_PRP Z-score standardization for PRP windows.
%
% Input:
%   X  : [nChan x nTime x nPRP]
% Output:
%   Xz : same size, z-scored across time (dim=2) for each channel and PRP

X = double(X);

if ndims(X) ~= 3
    error('Input must be 3-D: [nChan x nTime x nPRP].');
end

mu = mean(X, 2, 'omitnan');          % [nChan x 1 x nPRP]
sig = std(X, 0, 2, 'omitnan');       % [nChan x 1 x nPRP]
sig(sig == 0 | isnan(sig)) = 1;

Xz = (X - mu) ./ sig;                % implicit expansion
end