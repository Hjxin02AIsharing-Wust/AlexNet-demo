# AlexNet-demo
**This is our demo to reproduce AlexNet on pytorch**. Here's [my blog](https://www.cnblogs.com/hjxin02Aisharing-wust/p/17526775.html) on some interpretations of the AlexNet paper.

The AlexNet paper《**ImageNet Classification with Deep Convolutional Neural Networks**》,you can download it [here](https://dl.acm.org/doi/pdf/10.1145/3065386).
![image](https://github.com/Hjxin02AIsharing-Wust/AlexNet-demo/blob/main/image-texture/The%20Image%20Of%20AlexNet%20Network%20Structure.png)
The overall network structure of AlexNet includes: 1 input layer, 5 convolutional layers (C1, C2, C3, C4, C5), 2 fully connected layers (FC6, FC7) and 1 output layer.



## Dataset
The dataset uses more than 4000 images in 5 categories, you can download it [here](https://drive.google.com/drive/folders/1z2d7UejBR55QY8dc2GOmSkyfi8C-vUBs).

## Data Preprocess
You need to change the `root_file` parameter in the  `Data_Preprocess.py` file to the address of the dataset you downloaded. We follow the training set: validation set ratio of 9 to 1. You can also change this ratio, just change the `split_rate parameter`. Also we follow the data enhancement operation of the AlexNet paper, cropping the image to 227x227 at random and flipping it horizontally.

## Usage

### Train
You can use the following commands to train the model：
```shell
python train.py 
```
Here are some of our training settings: batchsize is set to 128, cross-entropy loss function is used, Adam is used for the optimizer, learning rate is set to 0.0002, and 10 epochs are trained.

## Test

We provide the test code：
```shell
python test.py
```
You can use our provided model weights [`AlexNet.pth`](https://drive.google.com/drive/folders/1uGx3-N93r2uie8_elHj_TqGk5muBhpkx) and test image `roseflower.png`, of course, you can also use your own.
This is our test result.




