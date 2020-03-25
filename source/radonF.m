function Ax = radonF(x, theta, imgsize)

if any(size(x)==1) || any(size(x)~=imgsize)
    x = reshape(x, imgsize);
end

% Radon (image -> line)
Ax = radon(x,theta);