# Pytorch-Nuclei-Instance-Segmentation-with-Watershed
This is a Pytorch demo for nuclei instance segmentation.

This demo reproduce and improve the Deep Interval-Masker-Aware Networks (DIMAN) and Marker-controlled Watershed method. The original implementation of DIMAN is showed by [appiek](https://github.com/appiek/Nuclei_Segmentation_Experiments_Demo).

In this work, we keep the preprocessing matlab code and marker-controlled watershed code in original implementation. We changed the other parts to python and pytorch version.

The validation dataset is [TNBC](https://zenodo.org/record/1175282#.Xnk84G5uKhd) nuclei segmentation dataset. You can also use the dataset after our preprocessing, which can be downloaded [here](https://drive.google.com/file/d/16ajg19swFmvFqkH5sxsdoI3GX4aqjRB6/view?usp=sharing).

# Visualization

![image](https://github.com/flyingdingding/Pytorch-Nuclei-Instance-Segmentation-Demo/blob/master/demo_images/visual.png)

# Packages version
- Pytorch 1.3.1
- numpy 1.17.3
- pillow 6.2.1
- optparse 1.5.3
- scipy 1.3.1
- opencv 4.2.0
- SimpleITK
- skimage 0.16.2

# Quick start

1. Download the preprocessed data from [link](https://drive.google.com/file/d/16ajg19swFmvFqkH5sxsdoI3GX4aqjRB6/view?usp=sharing)
2. Train the model
```
python train.py
```

