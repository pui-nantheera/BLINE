function [pleuralLine, indi, indj, confidence] = findPleuralLine(imR, thetaHor, optionDim, maxplueral, ratio, mean_ind)

% dimenstion
[h w] = size(imR);

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
    if sum(possiblearea(:))>0
        minrho = find(sum(possiblearea,2)>0);
        minrho = minrho(1);
        dimmask = (size(qHor,1)-minrho:-1:1);
        dimmask = [dimmask(1)*ones(1,minrho) dimmask]';
        dimmask = repmat(dimmask, [1 size(qHor,2)]);
        qHor = qHor.*dimmask;
    end
end

% find pleural line from the max of qHor
BW = imregionalmax(qHor);
%BW(1:min(size(qHor,1),round(1.5*size(qHor,2))),:) = 0;
if mean_ind(1) > 0
    limgap = 20;
    BW([1:round(mean_ind(1))-limgap round(mean_ind(1))+limgap:end],:) = 0;
    BW(:,[1:round(mean_ind(2))-limgap round(mean_ind(2))+limgap:end]) = 0;
    while sum(BW(:))<=0
        BW = imregionalmax(qHor);
        BW(1:min(size(qHor,1),round(1.5*size(qHor,2))),:) = 0;
        limgap = limgap*2;
        BW([1:round(mean_ind(1))-limgap round(mean_ind(1))+limgap:end],:) = 0;
        BW(:,[1:round(mean_ind(2))-limgap round(mean_ind(2))+limgap:end]) = 0;
    end
end
[indi, indj] = find(BW);
qHorMax = qHor(BW);
[val, inds] = sort(qHorMax,'descend');
inds = inds(1:min(length(inds),maxplueral));
indi = indi(inds(1));
indj = indj(inds(1));

tempR = zeros(size(qHor));
tempR(indi,indj) = max(qHor(:));
sidemax = max(size(imR));
pleuralLine = radonT(tempR, thetaHor, sidemax);
if size(pleuralLine,2) < w
    pleuralLine = padarray(pleuralLine,[round((w-size(pleuralLine,2))/2) round((w-size(pleuralLine,2))/2)]);
end
pleuralLine = pleuralLine(round(sidemax/2 - h/2) + (1:h),:);

confidence = val(1);%/sum(val(1:min(length(inds),maxplueral)));