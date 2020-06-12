
#!/bin/bash
set -e
### All of your tmp data will be saved in ./tmp folder

echo "Hello! I will prepare training data and starting to training step by step."

# 1. checking dataset if OK
if [ ! -d "./dataset/traindata" ]; then
	echo "Error: The traindata is not exist. Read dataset/README.md to get useful info."
	exit
fi

echo "Checking dataset pass."
if [ -d "./tmp" ]; then
	echo "Warning: The tmp folder is not empty. A good idea is to run ./clearAll.sh to clear it before training."
fi

# 2. stage: P-Net
echo "Preparing annotation file for training data"
python -u prepare_data/gen_anno_file.py
### generate training data(Face Detection Part) for PNet
echo "Preparing P-Net training data: bbox"
python -u prepare_data/gen_hard_bbox_pnet.py --mydata=True --lmnum=4
### generate training data(Face Landmark Detection Part) for PNet
echo "Preparing P-Net training data: landmark"
python -u prepare_data/gen_landmark_aug.py --stage=pnet --mydata=True --lmnum=4
### generate tfrecord file for tf training
echo "Preparing P-Net tfrecord file"
python -u prepare_data/gen_tfrecords.py --stage=pnet --lmnum=4
### start to training P-Net
echo "Start to training P-Net"
python -u training/train_plate.py --stage=pnet

# 3. stage: R-Net
### generate training data(Face Detection Part) for RNet
echo "Preparing R-Net training data: bbox"
python -u prepare_data/gen_hard_bbox_rnet_onet.py --stage=rnet --mydata=True --lmnum=4
### generate training data(Face Landmark Detection Part) for RNet
echo "Preparing R-Net training data: landmark"
python -u prepare_data/gen_landmark_aug.py --stage=rnet --mydata=True --lmnum=4
### generate tfrecord file for tf training
echo "Preparing R-Net tfrecord file"
python -u prepare_data/gen_tfrecords.py --stage=rnet --lmnum=4
### start to training R-Net
echo "Start to training R-Net"
python -u training/train_plate.py --stage=rnet

# 4. stage: O-Net
### generate training data(Face Detection Part) for ONet
echo "Preparing O-Net training data: bbox"
python prepare_data/gen_hard_bbox_rnet_onet.py --stage=onet --mydata=True --lmnum=4
### generate training data(Face Landmark Detection Part) for ONet
echo "Preparing O-Net training data: landmark"
python -u prepare_data/gen_landmark_aug.py --stage=onet --mydata=True --lmnum=4
### generate tfrecord file for tf training
echo "Preparing O-Net tfrecord file"
python -u prepare_data/gen_tfrecords.py --stage=onet --lmnum=4
### start to training O-Net
echo "Start to training O-Net"
python -u training/train_plate.py --stage=onet

# 5. Done
echo "Congratulation! All stages had been done. Now you can going to testing and hope you enjoy your result."
echo "haha...bye bye"
