% Copyright (c) Nantheera Anantrasirichai and Alin Achim
%
% This code is distributed under the terms of the GNU General Public License 3.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
warning off
addpath('./source');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RUNALLSCANS = 0;
DIMPLEURAL = 1;
thetaVer = -15:0.25:15;     % to detect vertical lines
thetaHor = (-5:0.25:5)+90;              % to detect horizontal lines
theraPleural = (-15:0.25:15)+90;        % to detect pleural lines
thetaDisplay = -45:0.5:135;
optdeblur = 0;
runDetectBLines = 1;


outputDir = 'results\';
mkdir(outputDir);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imgnoise = im2double(imread('3 b-lines.BMP'));


% dimentions
[horig, worig, totalSlices] = size(imgnoise);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% detect pleural line
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
runDetectPleuralLine;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lung space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
runDefineLungSpace;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% solving
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lungLines = [];
lungRadon = [];
lungDisplay = [];
radonVolDisp = [];
for slicenum = 1:totalSlices
    disp(['slicenum = ',num2str(slicenum)]);
    
    % radon transform
    imRadon = radon(imRAll(:,:,slicenum), thetaDisplay);
    imRadonOrig = imresize(imRadon, max(size(imRadon))*[1 1]);
    % variables
    hLung = [1 1 hw];
    HPposy = indic;
    lungSpace = imRAll(:,:,slicenum);
    Pline = pleuralLine;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % solved with inverse problem
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    runAngleOption = 0;   % run vertical and horizonral line simultaneously
    if runAngleOption==1
        theta = thetaHor;
        thetaList{1} = theta;
    elseif runAngleOption==2
        theta = thetaVer;
        thetaList{1} = theta;
    else
        theta = [thetaVer thetaHor];
        thetaList{1} = thetaVer;
        thetaList{2} = thetaHor;
    end
    
    MAX_ITER = 10;
    MIN_ITER = 2;
    tolA = 1e-3;
    if optdeblur
        alpha = 0.01;
        lambdagain = 0.001;
        p = 0.8;
    else
        alpha = 0.1;%0.05;           % 0.01 = default (higher is faster)
        lambdagain = 0.001;%1e-5;     % 0.001 = default (higher is sharper)
        p = 0.5;
    end
    q = p;
    pdfs = 9;
    normOption = 0;
    
    % hough via radon transform
    A = @(x) radonT(x,theta);    % (line -> image)
    At = @(x) radon(x,theta);    % (image -> line)
    Phi = @(x) norm(x(:),q)^q;  % lp norm
    if q<1
        Psi = @(x,T) solve_Lp(x,T, q);
    else
        Psi = @(x,T) softshrinkage(x,T); % soft threshold
    end
    
    % input noisy image
    R = lungSpace;
    if size(R,1)~=size(R,2)
        R = imresize(R, hLung(3)*[1 1]);
    end
    % dim top part - remove effect of pleural line
    if DIMPLEURAL
        weightmap = repmat([20:-1:1]', [1 size(R,2)]);
        weightmap = [weightmap; zeros(size(R,1)-size(weightmap,1), size(R,2))];
        weightmap = 1 - weightmap/max(weightmap(:));
        R = R.*weightmap;
    end
    R = R/max(R(:));
    
    % estimate pdf
    psfsize = 3;
    [~, psf] = estimatePSF(imgnoise(:,:,round(totalSlices/2)), [psfsize psfsize]);
    
    hw = hLung(3);
    K  = psf2otf(psf,[hw,hw]);
    KC = conj(K);
    H  = @(x) real(ifft2(K.*fft2(x)));
    HT = @(x) real(ifft2(KC.*fft2(x)));
    
    % dimension and parameters
    lambda_max = norm(R, 'inf');
    lambda = lambdagain*min(lambda_max,sum(sum(R.^2)));
    
    % regularization parameter
    tau = alpha*max(abs(R(:))); % higher is reduce more noise
    
    % ADMM parameter
    rhoADMM = 1;
    inv_u = 0.5*inv((real(ifft2(KC.*K)) + rhoADMM.*eye(size(K)) + 10^-10));
    
    % common value
    Rx = At(R);
    HTR = HT(R);
    
    % find pdf in radon domain
    if optdeblur
        psfsize = 5;
        imwrite(Rx/max(Rx(:)), 'temp/Rxf_blur.png','png');
        % using Shan's method
        delete('temp/Rxf_blur_kernel.png');
        while ~exist('temp/Rxf_blur_kernel.png','file')
            command = ['deblur temp/Rxf_blur.png temp/tempx.png ',num2str(psfsize), ' ',num2str(psfsize), ' 0.1 0.2'];
            dos(command);
        end
        psfx = im2double(imread('temp/Rxf_blur_kernel.png'));
        psfx = psfx(:,:,1)/sum(sum(psfx(:,:,1)));
        
        KD  = psf2otf(psfx,size(Rx));
        KCD = conj(KD);
        D  = @(x) real(ifft2(KD.*fft2(x)));
        DT = @(x) real(ifft2(KCD.*fft2(x)));
    end
    
    % initial
    [height, width] = size(R);
    l = zeros(size(Rx));
    x = zeros(size(Rx));
    w = zeros(height,width);
    z1 = zeros(height,width);
    z2 = zeros(height,width);
    z3 = zeros(height,width);
    CDl = zeros(height,width);
    Ax  = zeros(height,width);
    
    % loop process
    for k = 1:MAX_ITER
        
        % display result
        finalR = A(x);
        finalR = finalR/(max(finalR(:))+1e-5);
        if size(finalR,1)~=hLung(3)
            finalR = imresize(finalR, [hLung(3) hLung(3)]);
        end
        
        imRadon = radon(finalR,thetaDisplay);
        imRadon = imresize(imRadon, max(size(imRadon))*[1 1]);
        
        % update u = 2(H'H + rho*I)^{-1} (2H'y + rho wk + rho Cxk - z1k - z2k)
        % -----------------------------------------------------------------
        u = inv_u*(2*HTR + rhoADMM*w + rhoADMM*Ax + rhoADMM*CDl - z1 - z2 - z3);
        maxu = max(max(u(20:end-20,20:end-20)));
        u = min(1,max(0,u/maxu));
        
        % reorder l-update
        % -----------------------------------------------------------------
        if optdeblur == 1
            blurRf = At(u);
            blurRf = blurRf/max(blurRf(:));
            reconxAll = deblurInRadon(blurRf,thetaList,'matlab',pdfs, normOption);
            x1 = A(reconxAll);
            u = (x1/max(x1(:)) + 0.5*u)/1.5; % force gradually change
        end
        
        % update w = prox_lamda (u + z/rho)
        % -----------------------------------------------------------------
        w = prox_lp(u + z1/rhoADMM,p,lambda);
        
        % update x = argmin_x ||x||_p^q + 0.5*||u^{k+1} - Cx + z2k/rho||^2_2
        % -----------------------------------------------------------------
        qInput = At(u + z2/rhoADMM);
        tgain = 1;
        Ax = Inf; count = 1;
        while (sum(~isfinite(Ax(:))))&&(count<=5)
            Ax = solveRegularise(qInput,At,tau*tgain, ...
                'Debias',0,'AT', A, 'Phi', Phi, 'Psi', Psi, ...
                'Monotone',1,'Initialization',0,'StopCriterion',1,...
                'ToleranceA',tolA,'Verbose', 0, 'Iter_rate', max(tau,min(1,q/4)));
            tgain = tgain/2;
            count = count + 1;
        end
        if (sum(~isfinite(Ax(:))))
            Ax = u + z2/rhoADMM;
        end
        maxAx = max(max(Ax(20:end-20,20:end-20)));
        if maxAx == 0
            maxAx = max(Ax(:));
        end
        Ax = max(0,min(1,Ax/maxAx));
        x = At(Ax);
        x = x/max(x(:));
        
        % z-update
        % -----------------------------------------------------------------
        z1 = z1 + rhoADMM*(u - w);
        z2 = z2 + rhoADMM*(max(0,u - Ax));
        if (optdeblur == 1)
            l = x;
            CDl = A(D(l));
            CDl = CDl/max(CDl(:));
            z3 = z3 + rhoADMM*(max(0,u - CDl));
        end
        
        if (optdeblur == 1)
            HAx = H(A(D(l)));
            HADl = max(0,HAx/max(HAx(:)));
            objective(k) = 1/2*sum(sum((HADl - R).^2)) ...
                + lambda*sum(abs(w(:)).^p) + tau*norm(x(:),q)^q + tau*norm(l(:),1);
            if k > MIN_ITER
                if (objective(k) > objective(k-1)) || sum(abs(x(:)-prevx(:)))/sum(prevx(:)) < 1e-3
                    x = prevx;
                    break;
                end
            end
            prevx = l;
        else
            % compute objective cost
            HAx = H(A(x));
            HAx = max(0,HAx/max(HAx(:)));
            objective(k) = 1/2*sum(sum((HAx - R).^2)) ...
                + lambda*sum(abs(w(:)).^p) + tau*norm(x(:),q)^q;
            if k > MIN_ITER
                if (objective(k) > objective(k-1)) || sum(abs(x(:)-prevx(:)))/sum(prevx(:)) < 1e-3
                    x = prevx;
                    break;
                end
            end
            prevx = x;
        end
        
        if ~normOption
            xver = x(:,1:length(thetaVer));
            x(:,1:length(thetaVer)) = xver/max(xver(:));
            Ax = A(x);
            Ax = Ax/max(Ax(:));
        end
    end
    % display result
    finalR = A(x);
    finalR = finalR/max(finalR(:));
    
    
    imRadon = radon(finalR,thetaDisplay);
    % update variable
    lungLines = cat(3,lungLines, finalR);
    lungRadon = cat(3,lungRadon, imRadon);
    imRadonResized = imresize(imRadon, max(size(imRadon))*[1 1]);
    ratioResizeRadon = size(imRadonResized)./size(imRadon);
    lungRadonDisplay = imRadonResized/max(imRadonResized(:));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Detect B lines
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if runDetectBLines
        
        detectLines;
        displayResults;
        imwrite(lungDisplay(:,1:end/2,:), [outputDir, 'f_',num2str(slicenum),'.png']);
        
    end
end

