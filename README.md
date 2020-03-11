# TasselNetv2+

<p align="center">
  <img src="plant_counting.png" width="825"/>
</p>

This repository includes the official implementation of TasselNetv2+ for plant counting, presented in our submission:

**TasselNetv2+: A Fast Implementation for High-Throughput Plant Counting from High-Resolution RGB Imagery**

Frontiers in Plant Science, 2020, in submission

[Hao Lu](https://sites.google.com/site/poppinace/) and Zhiguo Cao


## Highlights
- **Highly Efficient:** TasselNetv2+ runs an order of magnitude faster than [TasselNetv2](https://link.springer.com/article/10.1186/s13007-019-0537-2) with around 30fps on image resolution of 1980×1080 on a single GTX 1070;
- **Effective:** It retrains the same level of counting accuracy compared to its counterpart TasselNetv2;


## Installation
The code has been tested on Python 3.7.4 and PyTorch 1.2.0. Please follow the official instructions to configure your environment. See other required packages in `requirements.txt`.

## Prepare Your Data
**Wheat Ears Counting**
1. Download the Wheat Ears Counting (WEC) dataset from: [Google Drive (2.5 GB)](https://drive.google.com/open?id=1XHcTqRWf-xD-WuBeJ0C9KfIN8ye6cnSs). I have reorganized the data, the credit of this dataset belongs to [this repository](https://github.com/simonMadec/Wheat-Ears-Detection-Dataset).
2. Unzip the dataset and move it into the `./data` folder, the path structure should look like this:
````
$./data/wheat_ears_counting_dataset
├──── train
│    ├──── images
│    └──── labels
├──── val
│    ├──── images
│    └──── labels
````

**Maize Tassels Counting**
1. Download the Maize Tassels Counting (MTC) dataset from: [Google Drive (1.8 GB)](https://drive.google.com/open?id=1IyGpYMS_6eClco2zpHKzW5QDUuZqfVFJ)
2. Unzip the dataset and move it into the `./data` folder, the path structure should look like this:
````
$./data/maize_counting_dataset
├──── trainval
│    ├──── images
│    └──── labels
├──── test
│    ├──── images
│    └──── labels
````

**Sorghum Heads Counting**
1. Download the Sorghum Heads Counting (SHC) dataset from: [Google Drive (152 MB)](https://drive.google.com/open?id=1msk8vYDyKdrYDq5zU1kKWOxfmgaXpy-P). The credit of this dataset belongs to [this repository](https://github.com/oceam/sorghum-head). I only use the two subsets that have dotted annotations available.
2. Unzip the dataset and move it into the `./data` folder, the path structure should look like this:
````
$./data/sorghum_head_counting_dataset
├──── original
│    ├──── dataset1
│    └──── dataset2
├──── labeled
│    ├──── dataset1
│    └──── dataset2
````

## Inference
Run the following command to reproduce our results of TasselNetv2+ on the WEC/MTC/SHC dataset:

    sh config/hl_wec_eval.sh
    
    sh config/hl_mtc_eval.sh
    
    sh config/hl_shc_eval.sh
    
- Results are saved in the path `./results/$dataset/$exp/$epoch`.
  
## Training
Run the following command to train TasselNetv2+ on the on the WEC/MTC/SHC dataset:

    sh config/hl_wec_train.sh
    
    sh config/hl_mtc_train.sh
    
    sh config/hl_shc_train.sh
    
    
## Play with Your Own Dataset
To use this framework on your own dataset, you may need to:
1. Annotate your data with dotted annotations. I recommend the [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/);
2. Generate train/validation list following the example in `gen_trainval_list.py`;
3. Write your dataloader following example codes in `hldataset.py`;
4. Compute the mean and standard deviation of RGB on the training set;
5. Create a new entry in the `dataset_list` in `hltrainval.py`;
6. Create a new `your_dataset.sh` following examples in `./config` and modify the hyper-parameters (e.g., batch size, crop size) if applicable.
7. Train and test your model. Happy playing:)

