# encoding=utf-8

import os,sys
import cv2,shutil
from collections import Counter
import json
import numpy as np
import warnings
from imgops.system.scandir import get_xmls
import copy
from imgops.anydraw.xml_ops import xmlread, xmlwrite


def forward(images, xmls):

    image = cv2.imread(images[0],1)
    H, W,_ = image.shape

    deltas_xy = [
        (0,0),(W//2,0),(0,H//2),(W//2,H//2)
    ]

    image_aug = np.zeros((H,W,3),np.uint8)
    aug_H, aug_W = H,W

    bboxes_aug = []
    N = min([4,len(images)])
    for k in range(N):
        image = cv2.imread(images[k], 1)



        H,W,_ = image.shape
        image = cv2.resize(image, (aug_W//2, aug_H//2),0,0,cv2.INTER_LINEAR)
        bboxes = xmlread(xmls[k])

        # for bbox in bboxes:
        #     (x0,y0),(x1,y1) = bbox['points']
        #     x0 = x0 // 2
        #     y0 = y0 // 2
        #     x1 = x1 // 2
        #     y1 = y1 // 2
        #     cv2.rectangle(image,(x0,y0),(x1,y1),(255,0,0),3)


        dx,dy = deltas_xy[k]

        sx, sy = image.shape[1] * 1.0 / W, image.shape[0] * 1.0 / H
        for bbox in bboxes:
            bbox_aug = copy.deepcopy(bbox)
            (x0,y0),(x1,y1) = bbox_aug['points']
            x0,y0 = int(x0 * sx) + dx, int(y0*sy) + dy
            x1,y1 = int(x1 * sx) + dx, int(y1*sy) + dy

            if x1 - x0 < 8 and y1 - y0 < 8:
                continue

            bbox_aug['points'] = ( (x0,y0),(x1,y1) )
            bboxes_aug.append(bbox_aug)

        image_aug[dy:dy + image.shape[0], dx:dx + image.shape[1], :] = copy.deepcopy(image)
        # for bbox in bboxes_aug:
        #     (x0,y0),(x1,y1) = bbox['points']
        #     cv2.rectangle(image_aug,(x0,y0),(x1,y1),(255,255,0),1)


        #cv2.imshow("vis",image_aug)
        #cv2.imshow("vis-0", image)
        #cv2.waitKey(-1)
    return (image_aug, bboxes_aug)



if __name__ == "__main__":
    images = ["g:\\_joint\\input\\1.jpg","g:\\_joint\\input\\2.jpg","g:\\_joint\\input\\2.jpg","g:\\_joint\\input\\1.jpg"]
    gts = ["g:\\_joint\\input\\1.xml","g:\\_joint\\input\\2.xml","g:\\_joint\\input\\2.xml","g:\\_joint\\input\\1.xml"]
    image, bboxes = forward(images,gts)
    jpg = f"g:\\_joint\\output\\{0}.jpg"
    cv2.imwrite(jpg,image)
    xml = f"g:\\_joint\\output\\{0}.xml"
    xmlwrite(xml,bboxes)







