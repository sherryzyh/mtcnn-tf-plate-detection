import numpy as np
import os
import cv2

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter * 1.0 / (box_area + area - inter)
    return ovr

def squared_im(img, size):
    """Padding rectangle plate images to square images

    Parameters:
    -----------
    img: tuple, shape(height, width, channel)
        input image
    size: int
        size of resized images
            pnet: 12
            rnet: 24
            onet: 48
    -----------

    Returns:
    -----------
    squared img: numpy array, shape(size, size, 3)
    """
    h, w, c = img.shape
    print("h %d w %d c %d" %(h,w,c))
    if w >= h:
        scale = size / float(w)
        resized_w = size
        resized_h = int(h * scale)
    else:
        scale = size / float(h)
        resized_h = size
        resized_w = int(w * scale)
    resized_im = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top = (size - resized_h)//2
    bottom = size - top - resized_h
    left = (size - resized_w)//2
    right = size - left - resized_w
    print("w", resized_w, "h",resized_h,"top %d bottom %d left %d right %d" %(top, bottom, left, right))
    resized_im = cv2.copyMakeBorder(resized_im,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
    return resized_im

def enlarge_det(bbox):
    """Enlarge the detected bounding box

    Parameters:
    ----------
    bbox: numpy array, shape n x 5
        input bbox
    ----------

    Returns:
    ----------
    enlarged_bbox: numpy array
    """
    enlarged_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1

    enlarged_bbox[:, 0] = bbox[:, 0] - w*0.1
    enlarged_bbox[:, 1] = bbox[:, 1] - h*0.1
    enlarged_bbox[:, 2] = bbox[:, 2] + w*0.1
    enlarged_bbox[:, 3] = bbox[:, 3] + h*0.1

    return enlarged_bbox

def convert_to_square(bbox):
    """Convert bbox to square

    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox

    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox

def getBboxLandmarkFromTxt(txt, lmnum, with_landmark=True):
    """
        Generate data from txt file
        return [(img_path, bbox, landmark)]
            bbox: [left, right, top, bottom]
            landmark: [(x1, y1), (x2, y2), ...]
    """
    dirname = os.path.dirname(txt)
    for line in open(txt, 'r'):
        line = line.strip()
        components = line.split(' ')
        img_path = os.path.join(dirname, components[0]) # file path

        if lmnum == 5: # face
            # bounding box, (x1, y1, x2, y2) reading from (x1, x2, y1, y2)
            bbox = (components[1], components[3], components[2], components[4])
        elif lmnum == 4: # plate
            # bounding box, (x1, y1, x2, y2) reading from (x1, y1, x2, y2)
            bbox = (components[1], components[2], components[3], components[4])        
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int, bbox))

        # landmark
        if not with_landmark:
            yield (img_path, BBox(bbox))
            continue
        landmark = np.zeros((lmnum, 2))
        for index in range(0, lmnum):
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv

        #normalize
        '''
        for index, one in enumerate(landmark):
            rv = ((one[0]-bbox[0])/(bbox[2]-bbox[0]), (one[1]-bbox[1])/(bbox[3]-bbox[1]))
            landmark[index] = rv
        '''
        yield (img_path, BBox(bbox), landmark)

class BBox(object):
    """
        Bounding Box of face
    """
    def __init__(self, bbox):
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[2]
        self.bottom = bbox[3]
        
        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)
    #offset
    def project(self, point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])
    #absolute position(image (left,top))
    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])
    #landmark: 5*2
    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p
    #change to offset according to bbox
    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p
    #f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    #self.w bounding-box width
    #self.h bounding-box height
    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])
