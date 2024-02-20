# encoding=utf-8

import os,sys
import cv2,shutil
from collections import Counter
import json
import numpy as np
import warnings
from imgops import math_iou
import random
import copy
from imgops.anydraw.xml_ops import xmlread, xmlwrite

def is_valid_bbox(cand, bboxes):
    (ax0, ay0),(ax1,ay1) = cand
    for bbox in bboxes:
        (bx0, by0), (bx1, by1) = bbox['points']
        if math_iou.calc_iou_rect((ax0,ax1,ay0,ay1),(bx0,bx1,by0,by1)) > 0:
            return False
    return True



def forward(images, xmls):

    image_aug = cv2.imread(images[0], 0)
    aug_H, aug_W = image_aug.shape

    bboxes_aug = xmlread(xmls[0])

    for k in range(1,len(images)):
        image = cv2.imread(images[k], 0)
        bboxes = xmlread(xmls[k])
        sx = random.choice([k / 10.0 for k in range(5,20)])
        sy = sx
        for bbox in bboxes:
            bbox_aug = copy.deepcopy(bbox)
            (x0,y0),(x1,y1) = bbox_aug['points']
            crop = cv2.resize(image[y0:y1,x0:x1],(0,0),sx,sy,cv2.INTER_LINEAR)
            crop_H, crop_W = crop.shape
            if crop_W < 8 and crop_H < 8:
                continue
            rot = cv2.getRotationMatrix2D((crop_W//2, crop_H//2), random.uniform(-10,10), 1.0)
            crop = cv2.warpAffine(crop, rot, (crop_W,crop_H),borderMode=cv2.BORDER_REFLECT_101)
            x0,y0 = random.choice([k for k in range(aug_W)]),random.choice([k for k in range(aug_H)])
            x1,y1 = x0 + crop_W, y0 + crop_H
            if y1 < aug_H and x1 < aug_W:
                flag = is_valid_bbox( ((x0,y0),(x1,y1)), bboxes_aug )
                if not flag:
                    continue
                bbox_aug['points'] = ( (x0,y0),(x1, y1) )
                bboxes_aug.append(bbox_aug)
                image_aug[y0:y1, x0:x1] = copy.deepcopy(crop)
    return (image_aug, bboxes_aug)



if __name__ == "__main__":
    images = ["g:\\_joint\\input\\2.jpg","g:\\_joint\\input\\1.jpg"]
    gts = ["g:\\_joint\\input\\2.xml","g:\\_joint\\input\\1.xml"]
    image, bboxes = forward(images,gts)
    jpg = f"g:\\_joint\\output\\{0}.jpg"
    cv2.imwrite(jpg,image)
    xml = f"g:\\_joint\\output\\{0}.xml"
    xmlwrite(xml,bboxes)







