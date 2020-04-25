% function detectLines

R = finalR;
if size(R,1)~=size(R,2)
    R = imresize(R, hLung(3)*[1 1]);
end
% dim top part - remove effect of pleural line
weightmap = repmat([round(min(2*HPposy,size(R,1)/4)):-1:1]', [1 size(R,2)]);
weightmap = [weightmap; zeros(size(R,1)-size(weightmap,1), size(R,2))];
weightmap = 1 - weightmap/max(weightmap(:));
R = R.*weightmap;
% dim edge effect
eth = 50;
weightmap = repmat([eth:-1:1], [size(R,1) 1]);
weightmap = [weightmap zeros(size(R,1), size(R,2)-2*size(weightmap,2)) weightmap(:,end:-1:1)];
weightmap = 1 - weightmap/max(weightmap(:));
R = R.*weightmap;
R = R/max(R(:));

% detect vertical line
% --------------------
Rver = radon(R, thetaVer);
% dimension
[h, w] = size(Rver);
linesmatV = zeros(h,w);
% local max calculation
if max(Rver(:))>0
    blurRf = imfilter(Rver/max(Rver(:)),fspecial('gaussian',round(15*min(1,hspace/wspace)),round(5*min(1,hspace/wspace))));
    blurRf = imdilate(blurRf,strel('disk',round(10*min(1,hspace/wspace))));
    localmx = imregionalmax(blurRf);
    stats = regionprops(localmx,'Centroid');
    indiV = zeros(length(stats),1);
    indjV = zeros(length(stats),1);
    for k = 1:length(stats)
        indiV(k) = round(stats(k).Centroid(2));
        indjV(k) = round(stats(k).Centroid(1));
    end
    % remove angle lines
    toremove = abs(thetaVer(indjV))>8;
    indiV(toremove) = [];
    indjV(toremove) = [];
    % remove weak lines
    if length(indjV) > 3
        th = max(max(Rver((indjV-1)*h + indiV))*0.25,hLung(3)/16);
    else
        th = min(max(Rver((indjV-1)*h + indiV))*0.5,hLung(3)/16);
    end
    toremove = Rver((indjV-1)*h + indiV) < ceil(th);
    indiV(toremove) = [];
    indjV(toremove) = [];
    % display vertical lines
    linesmatV((indjV-1)*h + indiV) = 1;
    linesmatV = iradon(linesmatV, thetaVer);
    if size(linesmatV,1)>hLung(3)
        linesmatV = linesmatV(2:end-1,2:end-1);
    end
    % remove line that not go to the bottom of the image
    % and change to straight lines for curvelinear conversion
    [linelabel, numlabel] = bwlabel(linesmatV/max(linesmatV(:))>0.1);
    getpoints = [];
    for k = 1:numlabel
        curline = linelabel==k;
        halfmat = sum(curline(end/3:end,:).*R(end/3:end,:));
        if  sum(halfmat) > th/3
            [~, indmax] = max(halfmat);
            getpoints = [getpoints indmax];
            % linesmatV = linesmatV - curline;
        end
    end
    if ~isempty(getpoints)
        linesmatV(:,:) = 0;
        linesmatV(:,getpoints) = 1;
    else
        linesmatV = repmat(linesmatV(100,:), [size(linesmatV,1) 1]);
    end
    % remove lines near edge
    linesmatV(:,[1:15 end-14:end]) = 0;
    % make lines thicker for visualisaion
    if max(linesmatV(:))>0
        linesmatV = bwareaopen(linesmatV/max(linesmatV(:))>0.1,size(linesmatV,1)/2);
        if max(linesmatV(:))>0
            linesmatV = imdilate(linesmatV/max(linesmatV(:)),strel('disk',1));
        end
    end
end

% detect A-lines
% --------------
Rhor = radon(R, thetaHor);
% dimension
[h, w] = size(Rhor);
% remove pleural line
[indiP, indjP] = find(Rhor == max(Rhor(:)));
linesmatP = zeros(h, w);
linesmatP(min(h,max(1,min(size(linesmatP,1),indiP + (-20:20)))), indjP) = 1;
linesmatP = imdilate(linesmatP, strel('rectangle',[10 100]));
linesmatP = imfilter(linesmatP, fspecial('gaussian', 50, 10));
linesmatP = linesmatP/max(linesmatP(:));
% a binary map of the possible areas of the A-lines
gA = HPposy;
if gA < size(R,1)
    wA = zeros(size(R));
    wA(gA:gA:end,:) = 1;
    wA = imdilate(wA,strel('disk',15));
    %wA(1:gA,:) = 1;
    xA = radon(wA, thetaHor);
    Rhor = Rhor.*(xA/max(xA(:))>0);
end

% local max calculation
if max(Rhor(:))> 0
    blurRf = imfilter(Rhor/max(Rhor(:))  - linesmatP,fspecial('gaussian',15,5));
    blurRf = imdilate(blurRf,strel('disk',4));
    localmx = imregionalmax(blurRf);
    % get points of A lines
    stats = regionprops(localmx,'Centroid');
    indiA = zeros(length(stats),1);
    indjA = zeros(length(stats),1);
    for k = 1:length(stats)
        indiA(k) = round(stats(k).Centroid(2));
        indjA(k) = round(stats(k).Centroid(1));
    end
    % remove pleural line
    [pleuralvalue,toremove] = max(Rhor((indjA-1)*h + indiA));
    [~,curtheta] = min(abs(thetaDisplay-thetaHor(indjA(toremove))));
    Pposx = curtheta(1)+10;
    Pposy = indiA(toremove);
%     indiA(toremove) = [];
%     indjA(toremove) = [];
    % remove weak line
    th = max(pleuralvalue/(10+optdeblur*4),max(Rhor((indjA-1)*h + indiA))*0.25);
    toremove = Rhor((indjA-1)*h + indiA) < th;
    indiA(toremove) = [];
    indjA(toremove) = [];
%     % remove near border
%     toremove = (indjA<w/10)|(indjA>9*w/10);
%     indiA(toremove) = [];
%     indjA(toremove) = [];
    % definitely B-lines
    indB = find(sum(R>0.05) > hLung(3)*0.75);
    
    maskBline = zeros(size(R));
    maskBline(:,indB) = 1;
    maskBline = imdilate(maskBline, strel('disk',5));
    
    if ~isempty(indjA)
        % vertical line width
        maskVer = imopen(R>0.01*(1+~optdeblur),strel('line',20,90));
        % display A-lines
        linematA = zeros(hLung(3),hLung(3));
        toremove = zeros(1,length(indjA));
        for k = 1:length(indjA)
            tempmat = zeros(h,w);
            tempmat(indiA(k),indjA(k)) = 1;
            tempmat = iradon(tempmat, thetaHor)>0;
            if size(tempmat,1)>hLung(3)
                tempmat = tempmat(2:end-1,2:end-1);
            end
            tempmat = tempmat.*R > 0.1*HPposy/70;
            tempmat = imclose(tempmat,strel('line',round(thetaVer(end)/2),0));
            if sum(tempmat(:).*wA(:))>0
                [tempmat,num] = bwlabel(tempmat);
                if num == 0
                    toremove(k) = 1;
                else
                    linevalue = zeros(1,num);
                    for m = 1:num
                        linevalue(m) = sum((tempmat(:)==m).*R(:));
                    end
                    [~, indm] = max(linevalue);
                    % find length of A
                    [indAi, indAj] = find(tempmat==indm);
                    lengthA = range(indAj);
                    if (lengthA>40)&&(sum(maskBline(tempmat(:)==indm))<15)
                        linematA = linematA + (tempmat==indm);
                    else
                        % remove points
                        toremove(k) = 1;
                    end
                end
            end
        end
        indiA(toremove>0) = [];
        indjA(toremove>0) = [];
        linematA = imdilate(linematA,strel('line',2,90));
        if sum(sum(linematA(1:gA+8,:))) > 0 % must have the first A line
            % remove Z-lines
            [vlabel, numV] = bwlabel(linesmatV>0);
            for k = 1:numV
                curVline = vlabel==k;
                [~,numCrossHor] = bwlabel(linematA.*curVline >0);
                curAlines = imreconstruct(linematA.*curVline >0, linematA>0);
                if (numCrossHor <= 1) && (sum(R(:).*curVline(:)) > sum(R(:).*curAlines(:)))
                    filinematA = (linematA - imdilate(curVline, strel('disk',10))) >0;
                else
                    % remove B-line if A-lines > 1
                    linesmatV = linesmatV - curVline;
                    linesmatV(linesmatV<0) = 0;
                    % remove points in radon transform
                    mapRadon = radon(curVline, thetaVer);
                    toremove = mapRadon((indjV-1)*h + indiV)>0;
                    indjV(toremove) = [];
                    indiV(toremove) = [];
                end
            end
        else
            linematA(:,:) = 0;
        end
    end
end
