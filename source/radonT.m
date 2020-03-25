function Ax = radonT(x, theta)

% inverse radon (line -> image)
Ax = iradon(x,theta);%,'Hamming'); % filtering if want super smooth
Ax = max(0,Ax(2:end-1,2:end-1));