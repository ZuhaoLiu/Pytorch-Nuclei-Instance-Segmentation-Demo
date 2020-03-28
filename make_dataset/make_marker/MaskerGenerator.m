function [Masker_mask, Masker_weight ]= MaskerGenerator(mask, scalar)
[H,W]=size(mask);
%% step1:距离变换
D = -bwdist(~mask);

%% step2:为避免过度分割，确定最小区域
Masker_mask = imextendedmin(D,scalar);

%% step3:计算权值
Num1 = sum(Masker_mask(:));
TotalNum = H*W;
Num2 = TotalNum - Num1;
Weight_class = Masker_mask.*(Num2/TotalNum) + (~Masker_mask).*(Num1/TotalNum);

Masker_weight = double(Weight_class + Masker_mask.*(1./(1+exp(D./10))));

