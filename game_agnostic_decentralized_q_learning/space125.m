function v = space125(a,b)
% SPACE125 create vectors with values 0.2, 0.5, 1, 2, 5, 10, 20, ...
% 
% Synopsis:
%   v = space125(a,b)
%      Creates a vector v with values 1*10^k, 2*10^k, and 5*10^k, so that
%      a <= x <= b for all x in v.
%
%      These values are usefull for creating logarithmic plots since they
%      are pretty regularly spaced in the log-space.
%%
% Filename     : space125.m
% Author       : Petr Posik (posik#labe.felk.cvut.cz, replace # by @ to get email)
% Created      : 22-Jan-2010 14:05:57
% Modified     : $Date: 2010-01-22 15:00:51 +0100 (pá, 22 I 2010) $
% Revision     : $Revision: 4599 $
% Developed in : 7.5.0.342 (R2007b)
% $Id: space125.m 4599 2010-01-22 14:00:51Z posik $
    % Determine minimal and maximal value of exponent
    minex = floor(log10(a));
    maxex = ceil(log10(b));
    
    % Create vector of 1,2,5 values for all exponents
    vals = [1 2 5]';
    exps = minex:maxex;
    v = bsxfun(@(x,y) x*10^y, vals, exps);
    v = v(:)';
    
    % Get rid of the out-of-bounds entries
    v = v(a <= v & v <= b);
end
