function Ax = radonF(x, theta, imgsize)

%
% Copyright (c) Nantheera Anantrasirichai and Alin Achim
%
% This code is distributed under the terms of the GNU General Public License 3.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if any(size(x)==1) || any(size(x)~=imgsize)
    x = reshape(x, imgsize);
end

% Radon (image -> line)
Ax = radon(x,theta);