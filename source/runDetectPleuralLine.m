% runDetectPleuralLine;

ratio = 0.5*0.75;
optionDim = 2;
maxplueral = 5;


h = min(horig,worig);
h = floor(h/2)*2;

boundplural = round(h/2);

pleuralLine = zeros(h,h,totalSlices);
meanImg = mean(imgnoise,3);
meanImg = meanImg(1:boundplural,:);
[curpLine, indi_m, indj_m, weight_m] = findPleuralLine(meanImg/max(meanImg(:)), theraPleural, optionDim, maxplueral,ratio,[0 0]);
mean_indi_p = indi_m;
mean_indj_p = indj_m;
indi_p = zeros(1,totalSlices);
indj_p = zeros(1,totalSlices);
weight_p = zeros(1,totalSlices);
gain = 2;
for k = 1:totalSlices
    %imR = imgnoise(1:h,1:h,k);
    imR = imgnoise(1:boundplural,:,k);
    [curpLine, indi_p(k), indj_p(k), weight_p(k)] = findPleuralLine(imR, theraPleural, optionDim, maxplueral,ratio,[mean_indi_p mean_indj_p]);
    if k > 10
        mean_indi_p = (indi_m*weight_m*gain + sum(indi_p.*weight_p))/(weight_m*gain + sum(weight_p));
        mean_indj_p = (indj_m*weight_m*gain + sum(indj_p.*weight_p))/(weight_m*gain + sum(weight_p));
    end
    
    [indic, indjc] = find(curpLine>0);
    indic = ceil(mean(indic(:)))+5;
    pleuralLine(1:size(curpLine,1),1:h,k) = curpLine(:,1:h);
    
    % draw box on the input image
    imgwithpline = repmat(imgnoise(:,:,k),[1 1 3]);
    imgwithpline(1:h,1:h,1) = imgwithpline(1:h,1:h,1) + pleuralLine(:,:,k);
    %imshow(imgwithpline);
end
%close('all')
%figure; imshow(imgwithpline); title(num2str(fnum));

% smooth surface
if totalSlices>1
    ind = find(pleuralLine(:)>25);
    [indi, indj, indk] = ind2sub(size(pleuralLine), ind);
    fsmooth = fit([indj, indk],indi,'poly23','Robust','Bisquare');
    plinesmth = round(fsmooth(indj, indk));
    ind = sub2ind(size(pleuralLine), plinesmth, indj, indk);
    pleuralLine(:,:,:) = 0;
    pleuralLine(ind) = 100;
end