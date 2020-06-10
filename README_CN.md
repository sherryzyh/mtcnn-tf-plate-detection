[English Version](./README.md)

## 项目简介

[MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/): Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks的TensorFlow 实现版本，在仓库 [mtcnn_tf](https://github.com/BobLiu20/mtcnn_tf) 实现人脸识别的基础上，改进以完成车牌的检测及识别。

本仓库可以进行人脸识别的训练及车牌识别的训练，人脸训练流程请参考[英文说明文档](./README.md)。中文版主要介绍车牌训练流程。

## 环境要求

* Ubuntu 14.04 or CentOS 7.2 及以上
* Cuda 8.0 及以上

## 依赖库

| 依赖库         | 版本     |
| -------------- | -------- |
| python         | 3.5      |
| tensorflow     | 1.15.0   |
| tensorflow-gpu | 1.15.0   |
| TF-slim        |          |
| opencv-python  | 4.2.0.34 |
| easydict       | 1.9      |

## 数据集准备

准备以[CCPD数据集](https://blog.csdn.net/yang_daxia/article/details/88234138_)格式命名的车牌数据。

1. 将训练集图片文件存放在`./dataset/traindata/`目录下。
2. 将测试集图片文件存在放`./testing/plates/`目录下。

## 模型训练

运行`./runMy.sh/`进行训练。

## 模型测试

运行`python testing/test_plate.py --stage=onet`进行测试，测试结果文件会存放在`./testing/results_onet/`目录下。

---

### 分步训练及测试

### 1# P-Net

准备训练集的annotation file ：`python -u prepare_data/gen_anno_file.py`

准备P-Net正/负/部分样本：`python -u prepare_data/gen_hard_bbox_pnet.py --mydata=True --lmnum=4`

准备P-Net关键点样本：`python -u prepare_data/gen_landmark_aug.py --stage=pnet --mydata=True --lmnum=4`

准备P-Net tfrecord文件： `python -u prepare_data/gen_tfrecords.py --stage=pnet --lmnum=4`

训练P-Net：`python -u training/train_plate.py --stage=pnet`

测试P-Net模型：`python -u testing/test_plate.py --stage=pnet`，测试结果存放在`./testing/results_pnet/`目录下

### 2# R-Net

准备R-Net困难样本：`python -u prepare_data/gen_hard_bbox_rnet_onet.py --stage=rnet --mydata=True --lmnum=4`

准备R-Net关键点样本：`python -u prepare_data/gen_landmark_aug.py --stage=rnet --mydata=True --lmnum=4`

准备R-Net tfrecrod文件：`python -u prepare_data/gen_tfrecords.py --stage=rnet --lmnum=4`

训练R-Net：`python -u training/train_plate.py --stage=rnet`

测试R-Net模型：`python -u testing/test_plate.py --stage=rnet`，测试结果存放在`./testing/results_rnet/`目录下

### 3# O-Net

准备O-Net困难样本：`python -u prepare_data/gen_hard_bbox_rnet_onet.py --stage=onet --mydata=True --lmnum=4`

准备O-Net关键点样本：`python -u prepare_data/gen_landmark_aug.py --stage=onet --mydata=True --lmnum=4`

准备O-Net tfrecrod文件：`python -u prepare_data/gen_tfrecords.py --stage=onet --lmnum=4`

训练O-Net：`python -u training/train_plate.py --stage=onet`

测试O-Net模型：`python -u testing/test_plate.py --stage=onet`，测试结果存放在`./testing/results_onet/`目录下