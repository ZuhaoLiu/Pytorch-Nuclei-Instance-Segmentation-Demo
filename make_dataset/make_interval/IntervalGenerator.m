function [Interval_mask, Interval_weight ]= IntervalGenerator(mask, threshold)
if nargin<2
    threshold = 0.35;
end
%% step1: Get edge
mask_contour = bwperim(mask);
%% step2: Get Connected region
[H,W]=size(mask);
[imLabel, num]= bwlabel(mask_contour);    %¶Ô¸÷Á¬Í¨Óò½øÐÐ±ê¼Ç  
X = repmat([1:H]',1,W);
Y = repmat([1:W],H,1);
%% step3: Calculate the distance from the point to the boundary
DistContour = zeros(H,W,num);
parfor i=1:num
    [Idx,Idy]=find(imLabel==i);
    nump = length(Idx);
    Idx = reshape(Idx,[1,1,nump]);
    Idy = reshape(Idy,[1,1,nump]);
    Dist = (repmat(X,1,1,nump)-repmat(Idx,H,W)).^2+(repmat(Y,1,1,nump)-repmat(Idy,H,W)).^2;
    Dist = (min(Dist,[],3)).^(0.5);
    DistContour(:,:,i) = Dist;
end
DistOrder = sort(DistContour,3);
DistFirst = DistOrder(:,:,1);
try 
    DistSecond = DistOrder(:,:,2);
    Interval_spatial = exp(-DistSecond./5);
catch
    Interval_spatial = zeros(size(mask));
end

%% step4: combine mask
Interval_mask = (Interval_spatial >= threshold) & (~mask);
%% step5:  compute weight
Num1 = sum(Interval_mask(:));
TotalNum = H*W;
Num2 = TotalNum - Num1;
Weight_class = Interval_mask.*(Num2/TotalNum) + (~Interval_mask).*(Num1/TotalNum);

Interval_weight =  Interval_spatial + Weight_class;