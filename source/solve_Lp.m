function   x   =  solve_Lp( y, lambda, p )

% iterative thresholding for p < 1
% if p >= 1, Newton's method is applied. 
maxy = max(y(:));
[x ,tau] = prox_lp(y/maxy,p,lambda);

% Shrinkage
x = abs(x)./(abs(x)+tau) .* y;
