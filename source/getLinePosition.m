function [indi, indj] = getLinePosition(Rf, rho, height, th, optionplot)%,theta)

if nargin < 4
    th = 40;
end
if nargin < 5
    optionplot = 0;
end
[hR, wR] = size(Rf);
blurRf = imfilter(Rf/max(Rf(:)),fspecial('gaussian',15,5));
blurRf = imdilate(blurRf,strel('disk',4));
localmx = imregionalmax(blurRf);
stats = regionprops(localmx,'Centroid');
indi = zeros(length(stats),1);
indj = zeros(length(stats),1);
for k = 1:length(stats)
    indi(k) = round(stats(k).Centroid(2));
    indj(k) = round(stats(k).Centroid(1));
end

% remove detected lines close to borders
vvalue = Rf((indj-1)*hR + indi);
maxv = max(vvalue);
toignore = (rho(indi) > height/2 - 20)|(vvalue < maxv*0.3);
indi(toignore) = [];
indj(toignore) = [];

% remove too close
vvalue = Rf((indj-1)*hR + indi);
toremove = zeros(1,length(indi));
for k = 1:length(indi)
    for m = k+1:length(indi)
        dist(k,m) = sqrt((indi(k)-indi(m))^2 + (indj(k)-indj(m))^2);
        if dist(k,m) < th
            if vvalue(k) < vvalue(m)
                toremove(k) = 1;
            else
                toremove(m) = 1;
            end
        end
    end
end
indi(toremove>0) = [];
indj(toremove>0) = [];

if optionplot
%     [xx,yy] = meshgrid(theta,rho);
%     figure; surf(xx, yy, Rf); title('R');
%     xlabel('\theta'); ylabel('rho');
%     hold on
%     for ii = 1:length(indi)
%         plot3(xx(indi(ii),indj(ii)),yy(indi(ii),indj(ii)),Rf(indi(ii),indj(ii)),'or')
%     end
    figure; imshow(Rf/max(Rf(:))); hold on
    plot(indj, indi, 'xb');
    for k = 1:length(indj)
        text(indj(k), indi(k), num2str(k),'color','red');
    end
end