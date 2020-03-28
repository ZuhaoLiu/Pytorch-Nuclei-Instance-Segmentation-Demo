for i=0:716 % change to your own image number
    label_path=['label/',num2str(i),'.png'];
    mask = imread(label_path);
    [Interval_mask, Interval_weight]  = IntervalGenerator(mask,0.4);
    imwrite(Interval_mask,['interval/', num2str(i), '.png']);
    save(['iw/', num2str(i), '.mat'],'Interval_weight');
end
