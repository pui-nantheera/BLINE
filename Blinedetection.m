function varargout = Blinedetection(varargin)
% BLINEDETECTION MATLAB code for Blinedetection.fig
%      BLINEDETECTION, by itself, creates a new BLINEDETECTION or raises the existing
%      singleton*.
%
%      H = BLINEDETECTION returns the handle to a new BLINEDETECTION or the handle to
%      the existing singleton*.
%
%      BLINEDETECTION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BLINEDETECTION.M with the given input arguments.
%
%      BLINEDETECTION('Property','Value',...) creates a new BLINEDETECTION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Blinedetection_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Blinedetection_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Blinedetection

% Last Modified by GUIDE v2.5 05-Nov-2016 23:05:09

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @Blinedetection_OpeningFcn, ...
    'gui_OutputFcn',  @Blinedetection_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Blinedetection is made visible.
function Blinedetection_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Blinedetection (see VARARGIN)

% Choose default command line output for Blinedetection
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% get(gca);
% set(gca,'xcolor',get(gcf,'color'));
% set(gca,'ycolor',get(gcf,'color'));
% set(gca,'ytick',[]);
% set(gca,'xtick',[]);
global thetaVer thetaHor thetaDisplay
thetaVer = -5:0.25:5;                 % to detect vertical lines
thetaHor = (-20:0.25:20)+90;            % to detect horizontal lines
thetaDisplay = -45:0.5:135;


% UIWAIT makes Blinedetection wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Blinedetection_OutputFcn(~, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in selectimage.
function selectimage_Callback(hObject, eventdata, handles)
% hObject    handle to selectimage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% get image/images of colour chart
[filename, pathname] = uigetfile({'*.jpg;*.tif;*.png;*.bmp;*.gif;*.tiff','All Image Files';...
    '*.*','All Files' },'Select images', 'MultiSelect', 'off');
% input image
tempMat = im2double(imread([pathname,filename]));
if size(tempMat,3)>3
    tempMat(:,:,4:size(tempMat,3)) = [];
end
if size(tempMat,3)==3
    tempMat = rgb2gray(tempMat);
end
imgnoise = (tempMat-min(tempMat(:)))/range(tempMat(:));
handles.imgnoise = imgnoise;
% show input image on the axes box
axes(handles.axesInputImage); % Switches focus to this axes object.
imshow(imgnoise);
axes(handles.axesInputImage); % Switches focus to this axes object.
set(handles.axesInputImage,'xtick',[],'ytick',[]);  % Get rid of ticks.
set(handles.axesInputImage,'Visible','On');
set(handles.detectPleuralLine,'Enable','On');
% reset all buttons and texts
axes(handles.axesRadonTdomain); % Switches focus to this axes object.
imshow(0.973*ones(500,500));
set(handles.textRadonTdomain,'Enable','Off');
set(handles.textIteration,'Visible','Off');
set(handles.textIterNumber,'String','');
axes(handles.axesResult); % Switches focus to this axes object.
imshow(0.973*ones(500,400));
set(handles.textRunning,'Visible','Off');
set(handles.textStatus,'String','');
set(handles.editLungSpace,'Visible','Off');
set(handles.detectBlines,'Visible','Off');
set(handles.textRedPLine,'Visible','Off');
set(handles.startRun,'Enable','Off');
set(handles.optwithoutdeblur,'Enable','Off');
set(handles.optwithblur,'Enable','Off');
guidata(hObject, handles);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in detectPleuralLine.
function detectPleuralLine_Callback(hObject, eventdata, handles)
% hObject    handle to detectPleuralLine (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

addpath('./source');

global thetaVer thetaHor thetaDisplay;

% detect pleural line
% ---------------------------------------------------------------------
ratio = 0.5*0.75;
optionDim = 2;
maxplueral = 5;
[horig, w] = size(handles.imgnoise);
h = min(horig,w);
h = floor(h/2)*2;
imR = handles.imgnoise(1:h,1:h);
[pleuralLine, indi_p, indj_p] = findPleuralLine(imR, thetaHor, optionDim, maxplueral,ratio);
[indic, indjc] = find(pleuralLine>0);
indic = ceil(mean(indic(:)))+5;
pleuralLine = imdilate(pleuralLine,strel('disk',1));

% draw box on the input image
axes(handles.axesInputImage); % Switches focus to this axes object.
imgwithpline = repmat(handles.imgnoise,[1 1 3]);
imgwithpline(1:h,1:h,1) = imgwithpline(1:h,1:h,1) + pleuralLine;
imshow(imgwithpline);

% show radon transform
set(handles.textRadonTdomain,'Visible','On');
imRAll = handles.imgnoise(indic:end,:,:);
[horig, w] = size(imRAll);
hw = floor(min(horig,w)/2)*2;
imRAll = imRAll(1:hw,1:hw);
rectangle('Position', [1 indic hw hw], 'EdgeColor','g');
imRadon = radon(imRAll, thetaDisplay);
imRadon = imresize(imRadon, max(size(imRadon))*[1 1]);
axes(handles.axesRadonTdomain); % Switches focus to this axes object.
imshow(imRadon/max(imRadon(:)));
set(handles.axesRadonTdomain,'xtick',[],'ytick',[]);  % Get rid of ticks.
set(handles.axesRadonTdomain,'Visible','On');
set(handles.textRadonTdomain,'Enable','On');
set(handles.startRun,'Enable','On');
set(handles.optwithoutdeblur,'Enable','On');
set(handles.optwithblur,'Enable','On');
set(handles.editLungSpace,'Visible','On');
set(handles.textRedPLine,'Visible','On');

% update variables
handles.hLung = [1 1 hw];
handles.Pposy = indic;
handles.lungSpace = imRAll;
handles.lungLines = imRAll;
handles.lungRadon = radon(imRAll,[thetaVer thetaHor]);
handles.lungRadonDisplay = imRadon/max(imRadon(:));
handles.resultDisplay = imRAll;
handles.Pline = pleuralLine;
guidata(hObject, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in editLungSpace.
function editLungSpace_Callback(hObject, eventdata, handles)
% hObject    handle to editLungSpace (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global thetaHor thetaVer thetaDisplay;

set(handles.textDrawPleuralLine,'Visible','On');
axes(handles.axesInputImage); % Switches focus to this axes object.
imshow(handles.imgnoise); hold on;
rect = getrect;
plot([1 size(handles.imgnoise,2)],[rect(2) rect(2)]-1,'r'); hold off;
rect(2) = rect(2)+5;
rectangle('Position', rect, 'EdgeColor','g'); hold off
set(handles.textDrawPleuralLine,'Visible','Off');
handles.Pposy = round(rect(2));
sizeRec = round(min(rect(3:4)));
if rem(sizeRec,2)
    sizeRec = sizeRec-1;
end
handles.hLung = round([rect(1:2) sizeRec rect(3:4)]);
pleuralLine = zeros(size(handles.Pline));
pleuralLine(handles.Pposy,:) = 1;
pleuralLine = imdilate(pleuralLine,strel('disk',1));

% update radon transform domain
imRAll = handles.imgnoise(handles.Pposy:min(size(handles.imgnoise,1),handles.Pposy+handles.hLung(5)-1),...
                handles.hLung(1)+(0:handles.hLung(4)-1));
imRadon = radon(imresize(imRAll,handles.hLung(3)*[1 1]), thetaDisplay);
imRadon = imresize(imRadon, max(size(imRadon))*[1 1]);
axes(handles.axesRadonTdomain); % Switches focus to this axes object.
imshow(imRadon/max(imRadon(:)));
set(handles.axesRadonTdomain,'xtick',[],'ytick',[]);  % Get rid of ticks.
set(handles.axesRadonTdomain,'Visible','On');
set(handles.editLungSpace,'Visible','On');
% update variables
handles.lungSpace = imRAll;
handles.lungLines = imRAll;
handles.lungRadon = radon(imRAll,[thetaVer thetaHor]);
handles.lungRadonDisplay = imRadon/max(imRadon(:));
handles.resultDisplay = imRAll;
handles.Pline = pleuralLine;
guidata(hObject, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in startRun.
function startRun_Callback(hObject, eventdata, handles)
% hObject    handle to startRun (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.textRunning,'Visible','On');
set(handles.textIteration,'Visible','On');
set(handles.textStatus,'String','Initialising parameters...');
% parameters
global thetaVer thetaHor thetaDisplay;

runAngleOption = 0;                     % run vertical and horizonral line simultaneously
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
optdeblur = get(handles.optwithblur, 'Value');

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
R = handles.lungSpace;
if size(R,1)~=size(R,2)
    R = imresize(R, handles.hLung(3)*[1 1]);
end
R = R/max(R(:));

% estimate pdf
set(handles.textStatus,'String','Estimating psf...');
psfsize = 3;
initpsf = ones(psfsize,psfsize);
[J,psf] = deconvblind(handles.imgnoise, initpsf);

hw = handles.hLung(3);
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
    axes(handles.axesResult); % Switches focus to this axes object.
    imshow(handles.imgnoise); 
    
    set(handles.textStatus,'String','Estimating blur kernel...');
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

% display iteration number (textIteration)
set(handles.textIteration,'Enable','On');
set(handles.textIterNumber,'Enable','On');
% loop process
for k = 1:MAX_ITER
    
    set(handles.textIterNumber,'String',num2str(k-1));
    % display result
    finalR = A(x);
    finalR = finalR/(max(finalR(:))+1e-5);
    if size(finalR,1)~=handles.hLung(3)
        finalR = imresize(finalR, [handles.hLung(3) handles.hLung(3)]);
    end
    axes(handles.axesResult); % Switches focus to this axes object.
    y = handles.imgnoise;
    if length(handles.hLung)>3
        temp = imresize(finalR, [handles.hLung(5) handles.hLung(4)]);
        y(handles.Pposy-5 + (1:handles.hLung(5)),handles.hLung(1)+(0:handles.hLung(4)-1)) = temp;
        imshow(y); hold on
        rectangle('Position', [handles.hLung(1) handles.Pposy-2 handles.hLung(4) handles.hLung(5)], 'EdgeColor','g'); hold off
    else
        y(handles.Pposy-5 + (1:handles.hLung(3)),handles.hLung(1)+(0:handles.hLung(3)-1)) = finalR;
        imshow(y); hold on
        rectangle('Position', [handles.hLung(1) handles.Pposy-2 handles.hLung(3) handles.hLung(3)], 'EdgeColor','g'); hold off
    end
    imRadon = radon(finalR,thetaDisplay);
    imRadon = imresize(imRadon, max(size(imRadon))*[1 1]);
    axes(handles.axesRadonTdomain); % Switches focus to this axes object.
    imshow(imRadon/max(imRadon(:)));
    pause(0.001);
    
    % update u = 2(H'H + rho*I)^{-1} (2H'y + rho wk + rho Cxk - z1k - z2k)
    % -----------------------------------------------------------------
    u = inv_u*(2*HTR + rhoADMM*w + rhoADMM*Ax + rhoADMM*CDl - z1 - z2 - z3);
    maxu = max(max(u(20:end-20,20:end-20)));
    u = min(1,max(0,u/maxu));
   
    % reorder l-update
    % -----------------------------------------------------------------
    if optdeblur == 1
        set(handles.textStatus,'String','Deblurring in Radon transform...'); pause(0.0001);
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
    set(handles.textStatus,'String','Processing Regularisation...'); pause(0.0001);
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
    Ax = Ax/maxAx;
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
    
    set(handles.textStatus,'String','Calculating objective cost...');
    if (optdeblur == 1)
        HAx = H(A(D(l)));
        HADl = max(0,HAx/max(HAx(:)));
        objective(k) = 1/2*sum(sum((HADl - R).^2)) ...
            + lambda*sum(abs(w(:)).^p) + tau*norm(x(:),q)^q + tau*norm(l(:),1);
        disp(['k = ',num2str(k)]);
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
        disp(['k = ',num2str(k)]);
        if k > MIN_ITER
            if (objective(k) > objective(k-1)) || sum(abs(x(:)-prevx(:)))/sum(prevx(:)) < 1e-3
                x = prevx;
                break;
            end
        end
        prevx = x;
    end
    set(handles.textStatus,'String',['Finished iteration ',num2str(k)]);
end
if ~normOption
    xver = x(:,1:length(thetaVer));
    x(:,1:length(thetaVer)) = xver/max(xver(:));
end
set(handles.textStatus,'String','Finished!');
% display result
finalR = A(x);
finalR = finalR/max(finalR(:));
axes(handles.axesResult); % Switches focus to this axes object.
y = handles.imgnoise;
if length(handles.hLung)>3
    temp = imresize(finalR, [handles.hLung(5) handles.hLung(4)]);
    y(handles.Pposy-5 + (1:handles.hLung(5)),handles.hLung(1)+(0:handles.hLung(4)-1)) = temp;
    imshow(y); hold on
    rectangle('Position', [handles.hLung(1) handles.Pposy-2 handles.hLung(4) handles.hLung(5)], 'EdgeColor','g'); hold off
else
    y(handles.Pposy-5 + (1:handles.hLung(3)),handles.hLung(1)+(0:handles.hLung(3)-1)) = finalR;
    imshow(y); hold on
    rectangle('Position', [handles.hLung(1) handles.Pposy-2 handles.hLung(3) handles.hLung(3)], 'EdgeColor','g'); hold off
end
imRadon = radon(finalR,thetaDisplay);
imRadonResized = imresize(imRadon, max(size(imRadon))*[1 1]);
axes(handles.axesRadonTdomain); % Switches focus to this axes object.
imshow(imRadonResized/max(imRadonResized(:)));
set(handles.detectBlines,'Visible','On');
% update variable
handles.lungLines = finalR;
handles.lungRadon = imRadon;
handles.lungRadonDisplay = imRadonResized/max(imRadonResized(:));
handles.resultDisplay = y;
guidata(hObject, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in detectBlines.
function detectBlines_Callback(hObject, eventdata, handles)
% hObject    handle to detectBlines (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global thetaVer thetaHor thetaDisplay;

% clear display
axes(handles.axesRadonTdomain);
imshow(handles.lungRadonDisplay);
axes(handles.axesResult);
imshow(handles.resultDisplay);
pause(0.00001);

R = handles.lungLines;
if size(R,1)~=size(R,2)
    R = imresize(R, handles.hLung(3)*[1 1]);
end
optdeblur = get(handles.optwithblur, 'Value');

% detect vertical line
% --------------------
set(handles.textStatus,'String','Identifying vertical lines.');
Rver = radon(R, thetaVer);
% dimension
[h, w] = size(Rver);
% local max calculation
blurRf = imfilter(Rver/max(Rver(:)),fspecial('gaussian',15,5));
blurRf = imdilate(blurRf,strel('disk',10));
localmx = imregionalmax(blurRf);
stats = regionprops(localmx,'Centroid');
indiV = zeros(length(stats),1);
indjV = zeros(length(stats),1);
for k = 1:length(stats)
    indiV(k) = round(stats(k).Centroid(2));
    indjV(k) = round(stats(k).Centroid(1));
end
% remove weak lines
th = min(max(Rver((indjV-1)*h + indiV))*0.25,handles.hLung(3)/16);
toremove = Rver((indjV-1)*h + indiV) < ceil(th);
indiV(toremove) = [];
indjV(toremove) = [];
% display vertical lines
linesmatV = zeros(h,w);
linesmatV((indjV-1)*h + indiV) = 1;
linesmatV = iradon(linesmatV, thetaVer);
if size(linesmatV,1)>handles.hLung(3)
    linesmatV = linesmatV(2:end-1,2:end-1);
end
linesmatV = imdilate(linesmatV/max(linesmatV(:)),strel('disk',1));

% detect A-lines
% --------------
set(handles.textStatus,'String','Identifying A-lines.');
Rhor = radon(R, thetaHor);
% dimension
[h, w] = size(Rhor);
% remove pleural line
[indiP, indjP] = find(Rhor == max(Rhor(:)));
linesmatP = zeros(h, w);
linesmatP(indiP + (-20:20), indjP) = 1;
linesmatP = imdilate(linesmatP, strel('rectangle',[10 100]));
linesmatP = imfilter(linesmatP, fspecial('gaussian', 50, 10));
linesmatP = linesmatP/max(linesmatP(:));
% a binary map of the possible areas of the A-lines
gA = handles.Pposy;
if gA < size(R,1)
    wA = zeros(size(R));
    wA(gA:gA:end,:) = 1;
    wA = imdilate(wA,strel('disk',15));
    wA(1:gA,:) = 1;
    xA = radon(wA, thetaHor);
    Rhor = Rhor.*(xA/max(xA(:))>0);
end
% local max calculation
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
indiA(toremove) = [];
indjA(toremove) = [];
% remove weak line
th = max(pleuralvalue/(10+optdeblur*4),max(Rhor((indjA-1)*h + indiA))*0.25);
toremove = Rhor((indjA-1)*h + indiA) < th;
indiA(toremove) = [];
indjA(toremove) = [];
% remove near border
toremove = (indjA<w/10)|(indjA>9*w/10);
indiA(toremove) = [];
indjA(toremove) = [];
% definitely B-lines
indB = find(sum(R>0.05) > handles.hLung(3)*0.75);
maskBline = zeros(size(R));
maskBline(:,indB) = 1;
maskBline = imdilate(maskBline, strel('disk',5));
if ~isempty(indjA)
    % vertical line width
    maskVer = imopen(R>0.01*(1+~optdeblur),strel('line',20,90));
    % display A-lines
    linematA = zeros(handles.hLung(3),handles.hLung(3));
    toremove = zeros(1,length(indjA));
    for k = 1:length(indjA)
        tempmat = zeros(h,w);
        tempmat(indiA(k),indjA(k)) = 1;
        tempmat = iradon(tempmat, thetaHor)>0;
        if size(tempmat,1)>handles.hLung(3)
            tempmat = tempmat(2:end-1,2:end-1);
        end
        tempmat = tempmat.*R > 0.01;
        tempmat = bwmorph(tempmat,'close');
        [tempmat,num] = bwlabel(tempmat);
        linevalue = zeros(1,num);
        for m = 1:num
            linevalue(m) = sum((tempmat(:)==m).*R(:));
        end
        [~, indm] = max(linevalue);
        % if length longer than width of the vertical line <PUI>
        onVerline = maskVer.*(tempmat==indm);
        if (sum(tempmat(:)==indm)/sum(onVerline(:)) > 3)&&(sum(maskBline(tempmat(:)==indm))<5)
            linematA = linematA + (tempmat==indm);
        else
            % remove points
            toremove(k) = 1;
        end
    end
    indiA(toremove>0) = [];
    indjA(toremove>0) = [];
    linematA = imdilate(linematA,strel('line',2,90));
    % remove Z-lines
    [vlabel, numV] = bwlabel(linesmatV>0);
    for k = 1:numV
        curVline = vlabel==k;
        [~,numCrossHor] = bwlabel(linematA.*curVline >0);
        if numCrossHor <= 1
            linematA = (linematA - imdilate(curVline, strel('disk',10))) >0;
        else
            linesmatV = linesmatV - curVline;
            linesmatV(linesmatV<0) = 0;
            % remove points in radon transform
            mapRadon = radon(curVline, thetaVer);
            toremove = mapRadon((indjV-1)*h + indiV)>0;
            indjV(toremove) = [];
            indiV(toremove) = [];
        end
    end
end
% display vertical lines and pleural line
axes(handles.axesRadonTdomain);
imshow(handles.lungRadonDisplay); hold on;
plot(Pposx, Pposy, 'xr');
if ~isempty(indjV)
    for k = 1:length(indjV)
        [~,curtheta] = min(abs(thetaDisplay-thetaVer(indjV(k))));
        plot(curtheta(1)+5, indiV(k), 'xg');
    end
end
if ~isempty(indjA)
    for k = 1:length(indjA)
        [~,curtheta] = min(abs(thetaDisplay-thetaHor(indjA(k))));
        plot(curtheta(1)+10, indiA(k), 'xb');
    end
end
hold off;
axes(handles.axesResult);
y = handles.imgnoise;

if length(handles.hLung)>3
    rangecol = handles.hLung(1)+(0:handles.hLung(4)-1);
    y(handles.Pposy-5 + (1:handles.hLung(5)),rangecol) = imresize(handles.lungLines, [handles.hLung(5) handles.hLung(4)]);
    y = repmat(y,[1 1 3]);
    y(handles.Pposy-5 + (1:handles.hLung(5)),rangecol,2) = y(handles.Pposy-5 + (1:handles.hLung(5)),rangecol,2) + ...
        imresize(linesmatV, [handles.hLung(5) handles.hLung(4)]);
else
    rangecol = handles.hLung(1)+(0:handles.hLung(3)-1);
    y(handles.Pposy-5 + (1:handles.hLung(3)),rangecol) = handles.lungLines;
    y = repmat(y,[1 1 3]);
    y(handles.Pposy-5 + (1:handles.hLung(3)),rangecol,2) = y(handles.Pposy-5 + (1:handles.hLung(3)),rangecol,2) + linesmatV;
end
hp = size(handles.Pline,1);
y(1:hp,1:hp,1) = y(1:hp,1:hp,1) + handles.Pline;
set(handles.textStatus,'String','Finished! Green lines are B-lines.');
if ~isempty(indjA)
    if length(handles.hLung)>3
        y(handles.Pposy-5 + (1:handles.hLung(5)),rangecol,3) = y(handles.Pposy-5 + (1:handles.hLung(5)),rangecol,3) + ...
            imresize(linematA, [handles.hLung(5) handles.hLung(4)]);
    else
        y(handles.Pposy-5 + (1:handles.hLung(3)),rangecol,3) = y(handles.Pposy-5 + (1:handles.hLung(3)),rangecol,3) + linematA;
    end
end
imshow(y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in optwithoutdeblur.
function optwithoutdeblur_Callback(hObject, eventdata, handles)
% hObject    handle to optwithoutdeblur (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.optwithblur, 'Value', 0);
% Hint: get(hObject,'Value') returns toggle state of optwithoutdeblur

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in optwithblur.
function optwithblur_Callback(hObject, eventdata, handles)
% hObject    handle to optwithblur (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.optwithoutdeblur, 'Value', 0);
% Hint: get(hObject,'Value') returns toggle state of optwithblur
