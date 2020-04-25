% runDefineLungSpace;

[indic, indjc] = find(sum(pleuralLine,3)>0.5);
indic = ceil(mean(indic(:)))+5;
imRAll = imgnoise(indic:end,:,:);
% crop to rectangular
[hspace, wspace, d] = size(imRAll);
if rem(hspace,2)
    imRAll(end+1,:,:) = imRAll(end,:,:);
end
if rem(wspace,2)
    imRAll(:,end+1,:) = imRAll(:,end,:);
end
[hspace, wspace, d] = size(imRAll);
if wspace>hspace
    % resize to square
    imRAll = imresize(imRAll,[hspace hspace]);
    hw = hspace;
elseif wspace<hspace
    hw = floor(wspace/2)*2;
    imRAll = imRAll(1:hw,1:hw,:);
end
HPposy = indic;
hLung = [1 1 hw];