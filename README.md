# Source code for DMRH-ICASSP2020[Pytorch Version]
## Introduction
### 1. Brief Introduction
This package contains the code for paper Deep Multi-Region Hashing on ICASSP2020. We carry out experiment on CIFAR-10, NUS-WIDE and MS-COCO datasets. We utilize the pre-trained CNN-F to initilize our network.
### 2. Running Environment
```
python3
pytorch
```
### 3. Datasets
we carry out experiment on CIFAR-10, NUS-WIDE and MS-COCO datasets. For NUS-WIDE dataset, please put all images into folder ./data/NUS-WIDE/Flickr/. For MS-COCO dataset, please put all training images into folder ./data/COCO2014/train2014/, all test images into folder ./data/COCO2014/val2014/. 
### 4. Run Demo
```
python train.py --gpu_id 0 --npatch 2 --dataset 0 --nbit 64 --is_train True
# dataset: {0: CIFAR10, 1: NUS-WIDE, 2: MS-COCO}
```
### 5. Results
#### 5.1 Top-5K Mean Average Precision on CIFAR-10
|N|24bits|48bits|64bits|128bits|
|:---|:---|:---|:---|:---|
|1|0.9019|0.8993|0.8983|0.8974|
|2|0.9101|0.9134|0.9099|0.9099|
|3|0.9178|0.9127|0.9169|0.9144|
|4|0.9172|0.9166|0.9179|0.9155|
|5|0.9216|0.9235|0.9241|0.9212|
|6|0.9256|0.9243|0.9245|0.9255|
|7|0.9249|0.9220|0.9233|0.9225|
|8|0.9243|0.9219|0.9229|0.9220|
|9|0.9174|0.9208|0.9182|0.9190|
|10|0.9149|0.9188|0.9190|0.9125|
#### 5.2 Top-5K Mean Average Precision on MS-COCO
|N|24bits|48bits|64bits|128bits|
|:---|:---|:---|:---|:---|
|1|0.7053|0.7168|0.7224|0.7354|
|2|0.7197|0.7437|0.7470|0.7616|
|3|0.7286|0.7427|0.7450|0.7540|
|4|0.7348|0.7514|0.7540|0.7611|
|5|0.7402|0.7565|0.7637|0.7724|
|6|0.7439|0.7656|0.7723|0.7854|
|7|0.7483|0.7735|0.7783|0.7930|
|8|0.7497|0.7782|0.7839|0.7984|
|9|0.7497|0.7795|0.7876|0.8008|
|10|0.7510|0.7831|0.7890|0.8061|
