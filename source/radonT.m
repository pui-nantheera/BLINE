function Ax = radonT(x, theta, output_size)

%
% Copyright (c) Nantheera Anantrasirichai and Alin Achim
%
% This code is distributed under the terms of the GNU General Public License 3.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
    % inverse radon (line -> image)
    Ax = iradon(x,theta);%,'Hamming'); % filtering if want super smooth
    
else
    Ax = iradon(x,theta, output_size);
end
Ax = max(0,Ax(2:end-1,2:end-1));