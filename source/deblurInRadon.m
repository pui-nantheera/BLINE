function reconxAll = deblurInRadon(blurRfAll,thetaList,method, psfsize, normOption)

if nargin < 3
    mothod = 'matlab';
end

% parameters
regHWidth = 150;
if (nargin < 4)|| isempty(psfsize) || (psfsize==0)
    psfsize = 15;
end
if nargin < 5
    normOption = 0;
end
% stopping theshold
tolA = 1e-3;

% separage Rf matrix as thetaList
blurRfAllsub{1} = blurRfAll(:,1:length(thetaList{1}));
blurRfAllsub{2} = blurRfAll(:,length(thetaList{1})+1:end);

reconxAll = []; psf = 0;
for thetanum = 1:length(thetaList)
    reconx1{thetanum} = [];
    
    if ~isempty(thetaList{thetanum})
        theta = thetaList{thetanum};
        blurRf = blurRfAllsub{thetanum};
        if normOption
            blurRf = blurRf/max(blurRf(:));
        end
        
        % dimenstion
        [height, width] = size(blurRf);
        
        % tv parameters
        tv_iters = 5;
        Psi = @(x,th)  tvdenoise(x,1/th,tv_iters);
        Phi = @(x) TVnorm(x);
        
        
        % -------------------------------------------------------------------------
        % estimate psf
        % -------------------------------------------------------------------------
        % detect peak orienation and rho
        [~,rho] = radon(radonT(blurRf, theta),theta);
        [indi, indj] = getLinePosition(blurRf, rho, height, 55, 0);
        valueRf = blurRf((indj-1)*height + indi);
        [valueRf, indsort] = sort(valueRf,'descend');
        indsort = indsort(valueRf>0.5);
        indi = indi(indsort);
        indj = indj(indsort);
        
        wpatch = fspecial('gaussian',round(2*regHWidth)*2+1,round(regHWidth/2));
        wpatch = wpatch/max(wpatch(:));
        reconx = 0.01*blurRf;
        countmap = 0.01*ones(height,width);
        for k = 1:min(3,length(indi))
            
            % estimate psf
            rangei = unique(max(1,min(height,indi(k) + (-regHWidth:regHWidth))));
            rangej = unique(max(1,min(width,indj(k) + (-regHWidth:regHWidth))));
            blurRfROI = blurRf(rangei, rangej);
            
            if strcmpi(method,'matlab')
                initpsf = ones(psfsize,psfsize);
                [J,psf] = deconvblind(blurRfROI, initpsf);
            end
            if strcmpi(method,'Shan') || sum(psf(:))==0
                
                imwrite(blurRfROI, 'Rf_blur.png','png');
                % using Shan's method
                delete('Rf_blur_kernel.png');
                while ~exist('Rf_blur_kernel.png','file')
%                     disp(['Running k=',num2str(k)]);
                    command = ['deblur.exe Rxf_blur.png tempx.png ',num2str(psfsize), ' ',num2str(psfsize), ' 0.1 0.2'];
                    dos(command);
                end
                psf = im2double(imread('temp/Rf_blur_kernel.png'));
                psf = psf(:,:,1)/sum(sum(psf(:,:,1)));
            end
            
            % patch reconstruction
            [rangei,ic] = unique(max(1,min(height,indi(k) + (-2*regHWidth:2*regHWidth))));
            [rangej,jc] = unique(max(1,min(width,indj(k) + (-2*regHWidth:2*regHWidth))));
            blurRfROI = blurRf(rangei, rangej);
            curwpatch = wpatch(ic,jc);
            
            % dimension
            [hROI, wROI] = size(blurRfROI);
            
            % -------------------------------------------------------------------------
            % debluring
            % -------------------------------------------------------------------------
            % kernel
            K  = psf2otf(psf,[hROI, wROI]);
            KC = conj(K);
            H  = @(x) real(ifft2(K.*fft2(x)));
            HT = @(x) real(ifft2(KC.*fft2(x)));
            
            % regularization parameter
            alpha = 0.1;% 0.05;
            tau = alpha*max(abs(blurRfROI(:))); % higher is reduce more noise
            
            x = Inf; count = 1;tgain = 1;
            while ((sum(~isfinite(x(:))))||(sum(x(:))==0))&&(count<=5)
                x = solveRegularise(blurRfROI,H,tau*tgain, ...
                    'Debias',0,'AT', HT,'Phi', Phi, 'Psi', Psi, ...
                    'Monotone',1,'Initialization',0,'StopCriterion',1,...
                    'ToleranceA',tolA,'Verbose', 0);
                tgain = tgain/2;
                count = count + 1;
            end
            % adjust scaling intensity
            x = x/max(x(:))*(max(blurRfROI(:)));
            
            c = normxcorr2(x/max(x(:)), blurRfROI/max(blurRfROI(:)));
            [max_c, imax] = max(abs(c(:))); %find the greatest correlation
            [ypeak, xpeak] = ind2sub(size(c),imax(1)); %find the greatest correlation
            ypeak = size(x,1) - ypeak;
            xpeak = size(x,2) - xpeak;
            if ~((ypeak==0)&&(xpeak==0))
                x = circshift(x,-[ypeak xpeak]);
            end
            
            % replace patch into its part of the whole image
            rangeic = unique(max(1,min(height,indi(k) + (round(-1.5*regHWidth):round(1.5*regHWidth)))));
            rangejc = unique(max(1,min(width,indj(k) + (round(-1.5*regHWidth):round(1.5*regHWidth)))));
            rangeir = ismember(rangei,rangeic);
            rangejr = ismember(rangej,rangejc);
            reconx(rangeic,rangejc) = reconx(rangeic,rangejc) + curwpatch(rangeir,rangejr).*x(rangeir,rangejr);
            countmap(rangeic,rangejc) = countmap(rangeic,rangejc) + curwpatch(rangeir,rangejr);
            %     figure; imshow(x/max(x(:)))
            
        end
        % -------------------------------------------------------------------------
        reconx1{thetanum} = reconx./countmap;
        if normOption
            reconx1{thetanum} = reconx1{thetanum}/max(reconx1{thetanum}(:));
        end
        reconxAll = [reconxAll reconx1{thetanum}];
    end
end