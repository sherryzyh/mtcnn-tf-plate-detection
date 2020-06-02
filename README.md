## Description
[MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/): Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks    
Implement training and testing by tensorflow.
Referenced [mtcnn_tf](https://github.com/BobLiu20/mtcnn_tf)

## Dependencies
* Tensorflow v1.0.0 or higher
* TF-Slim
* Python 3.5
* Ubuntu 14.04 or CentOS 7.2 or higher
* Cuda 8.0 or higher

## Prepare Face Data and Go Through Training Process
1. WIDER face dataset: Download WIDER_train.zip from [here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). You can only download Wider Face Training Images. Unzip it and move it to `dataset` folder.
2. Landmark dataset: Download **train.zip** from [here](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm). You can only download [training set](http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip). Unzip it and move it to `dataset` folder.
3. Run `./clearAll.sh` to clear all tmp file.
4. Run `./runAll.sh` to finish all (include preparing data and training). Please check this shell script to get more info.


## Prepare Plate Data and Train

1. Prepare your own plate dataset, make sure images named by CCPD rules. Put image files in `./dataset/traindata` folder.
2. Run `./runMy.sh` to finish all (include preparing data and training).

## Testing and Predict
1. Copy your image file to `testing/images`
2. Run `python testing/test_images.py --stage=onet`. Anyway you can specify stage to pnet or rnet to check your model.
3. The result will output in `testing/results_onet`

## Testing and Predict Plate Data
1. Copy your image file to `testing/plates`
2. Run `python testing/test_plates.py --stage=onet`. Anyway you can specify stage to pnet or rnet to check your model.
3. The result will output in `testing/results_onet`




## Results

![result1.png](https://i.loli.net/2017/08/30/59a6b65b3f5e1.png)

![result2.png](https://i.loli.net/2017/08/30/59a6b6b4efcb1.png)

![result3.png](https://i.loli.net/2017/08/30/59a6b6f7c144d.png)

![reult4.png](https://i.loli.net/2017/08/30/59a6b72b38b09.png)

![result5.png](https://i.loli.net/2017/08/30/59a6b76445344.png)

![result6.png](https://i.loli.net/2017/08/30/59a6b79d5b9c7.png)

![result7.png](https://i.loli.net/2017/08/30/59a6b7d82b97c.png)

![result8.png](https://i.loli.net/2017/08/30/59a6b7ffad3e2.png)

![result9.png](https://i.loli.net/2017/08/30/59a6b843db715.png)

## License
MIT LICENSE

## References
1. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
2. [MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)

