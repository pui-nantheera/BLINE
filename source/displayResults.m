% displayResults;

% display vertical lines and pleural line
% -------------------------------------------------------------
%
% Copyright (c) Nantheera Anantrasirichai and Alin Achim
%
% This code is distributed under the terms of the GNU General Public License 3.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if exist('lungRadonDisplay','var')
radonmatP = zeros(size(lungRadonDisplay));
radonmatP(round(Pposy*ratioResizeRadon(1)),round(Pposx*ratioResizeRadon(2))) = 1;
radonmatP = imdilate(radonmatP, strel('disk',5));
lungRadonDisp = repmat(lungRadonDisplay,[1 1 3]);
lungRadonDispNolines = lungRadonDisp;
imRadonOrig = repmat(imRadonOrig/max(imRadonOrig(:)),[1 1 3]);
imRadonOrigNolines = imRadonOrig;
lungRadonDisp(:,:,1) = lungRadonDisp(:,:,1) + radonmatP;
imRadonOrig(:,:,1) = imRadonOrig(:,:,1) + radonmatP;
if ~isempty(indjV)
    radonmatV = zeros(size(lungRadonDisplay));
    for k = 1:length(indjV)
        [~,curtheta] = min(abs(thetaDisplay-thetaVer(indjV(k))));
        radonmatV(round(indiV(k)*ratioResizeRadon(1)),round(curtheta(1)*ratioResizeRadon(2))) = 1;
    end
    radonmatV = imdilate(radonmatV, strel('disk',5));
    lungRadonDisp(:,:,2) = lungRadonDisp(:,:,2) + radonmatV;
    imRadonOrig(:,:,2) = imRadonOrig(:,:,2) + radonmatV;
end
if ~isempty(indjA)
    radonmatA = zeros(size(lungRadonDisplay));
    for k = 1:length(indjA)
        [~,curtheta] = min(abs(thetaDisplay-thetaHor(indjA(k))));
        radonmatA(round(indiA(k)*ratioResizeRadon(1)),round(curtheta(1)*ratioResizeRadon(2))) = 1;
    end
    radonmatA = imdilate(radonmatA, strel('disk',5));
    lungRadonDisp(:,:,3) = lungRadonDisp(:,:,3) + radonmatA;
    imRadonOrig(:,:,3) = imRadonOrig(:,:,3) + radonmatA;
end
radonVolDisp = cat(4,radonVolDisp,[imRadonOrigNolines imRadonOrig lungRadonDisp lungRadonDispNolines]);
end

% display in imnoise
% -------------------------------------------------------------
y = imgnoise(:,:,slicenum);
y_orig = y;

rangei = HPposy + (1:hLung(3));
rangej = 1:size(finalR,2);
if length(hLung)>3
    rangecol = rangej;%hLung(1)+(0:hLung(4)-1);
    y(HPposy + (1:hLung(5)),rangecol) = imresize(finalR, [hLung(5) hLung(4)]);
    y = repmat(y,[1 1 3]);
    yNoLines = y;
    y(HPposy + (1:hLung(5)),rangecol,2) = y(HPposy + (1:hLung(5)),rangecol,2) + ...
        imresize(linesmatV, [hLung(5) hLung(4)]);
else
    %                 rangecol = rangej;%hLung(1)+(0:hLung(3)-1);
    % y(HPposy:end,:) = imresize(finalR,[size(y(HPposy:end
    y(rangei,rangej) = finalR;
    %                 y_orig = imresize(y_orig, size(y));
    y = repmat(y,[1 1 3]);
    yNoLines = y;
    y_orig = repmat(y_orig,[1 1 3]);
    y(HPposy:end,:,2) = y(HPposy:end,:,2) + imresize(linesmatV,size(y(HPposy:end,:,2)));
    y_orig(HPposy:end,:,2) = y_orig(HPposy:end,:,2) + imresize(linesmatV,size(y_orig(HPposy:end,:,2)));
end
hp = size(pleuralLine,1);
y(1:hp,1:hp,1) = y(1:hp,1:hp,1) + pleuralLine(:,:,slicenum);
y_orig(1:hp,1:hp,1) = y_orig(1:hp,1:hp,1) + pleuralLine(:,:,slicenum);

if ~isempty(indjA)
    if length(hLung)>3
        y(HPposy + (1:hLung(5)),rangecol,3) = y(HPposy + (1:hLung(5)),rangecol,3) + ...
            imresize(linematA, [hLung(5) hLung(4)]);
    else
        y(rangei,rangej,3) = y(rangei,rangej,3) + linematA;
        if size(y_orig,2) > size(linematA,2)
            if rangei(end) > size(y_orig,1)
                rangei_ = rangei;
                rangei_(rangei>size(y_orig,1)) = [];
                y_orig(rangei_,rangej,3) = y_orig(rangei_,rangej,3) + linematA(1:length(rangei_(:)),:);
            else
                y_orig(rangei,rangej,3) = y_orig(rangei,rangej,3) + linematA;
            end
        else
            y_orig(rangei,:,3) = y_orig(rangei,:,3) + linematA(:,1:size(y_orig,2));
        end
    end
end
newresult = [repmat(imgnoise(:,:,slicenum),[1 1 3]) y_orig y(1:size(y_orig,1),:,:) yNoLines(1:size(y_orig,1),:,:)];
lungDisplay = cat(4,lungDisplay,newresult);
