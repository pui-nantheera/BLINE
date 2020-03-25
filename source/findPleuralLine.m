function [pleuralLine, indi, indj] = findPleuralLine(imR, thetaHor, optionDim, maxplueral, ratio)

% dimenstion
h = size(imR,1);

% dim top part
if optionDim==1
    
    ratio = round(size(img,1)/size(img,2))+2;          % if optionDim = 1; whereabout is the top part
    %         if mean(imR(:))>0.5
    %         end
    dimmask = [1:round(horig/ratio) round(horig/ratio)*ones(1,length(round(horig/ratio)+1:h))]';
    dimmask = repmat(dimmask, [1 h]);
    imR = imR.*dimmask./max(dimmask(:));
end

% Radon transform
[qHor,rhoHor] = radon(imR, thetaHor);

if optionDim==2
    % possible area with pleural line present
    numpix = 0;
    while (numpix/numel(qHor) < 0.01)&&(ratio>0)
        possiblearea = qHor > h*ratio;
        numpix = sum(possiblearea(:));
        ratio = ratio - 0.1;
    end
    minrho = find(sum(possiblearea,2)>0);
    minrho = minrho(1);
    dimmask = (size(qHor,1)-minrho:-1:1);
    dimmask = [dimmask(1)*ones(1,minrho) dimmask]';
    dimmask = repmat(dimmask, [1 size(qHor,2)]);
    qHor = qHor.*dimmask;
end

% find pleural line from the max of qHor
BW = imregionalmax(qHor);
[indi, indj] = find(BW);
qHorMax = qHor(BW);
[val, inds] = sort(qHorMax,'descend');
inds = inds(1:maxplueral);
indi = indi(inds(1));
indj = indj(inds(1));

tempR = zeros(size(qHor));
tempR(indi,indj) = max(qHor(:));
pleuralLine = radonT(tempR, thetaHor);