import os
import numpy as np

im_dir = "./dataset/traindata/"
anno_file = "./dataset/anno_file.txt"
f = open(anno_file,'w')

im_list = os.listdir(im_dir)
for im_name in im_list:
    #print(im_name)
    labels = im_name.split('-')
    box = labels[2].split('_')
    for i in range(len(box)):
        box[i] = box[i].split('&')
    lm = labels[3].split('_')
    for i in range(len(lm)):
        lm[i] = lm[i].split('&')
    box = np.array(box).reshape(-1)
    lm = np.array(lm).reshape(-1)
    anno = "%s %s %s %s %s %s %s %s %s %s %s %s %s\n" %("traindata/" + im_name,box[0],box[1],box[2],box[3],lm[0],lm[1],lm[2],lm[3],lm[4],lm[5],lm[6],lm[7])
    f.write(anno)

f.close()
