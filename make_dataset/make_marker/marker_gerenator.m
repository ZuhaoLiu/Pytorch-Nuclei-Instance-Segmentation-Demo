for i=0:716 % change to your own image number
    label_path=['label/',num2str(i),'.png'];
    mask = imread(label_path);
    [Masker_mask, Masker_weight] = MaskerGenerator(mask,2);
    imwrite(Masker_mask,['marker/', num2str(i), '.png']);
    save(['mw/', num2str(i), '.mat'],'Masker_weight');
end
