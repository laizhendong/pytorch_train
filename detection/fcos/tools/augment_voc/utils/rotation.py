

import cv2
import numpy as np
import copy,random
from imgops.anydraw.xml_ops import xmlread

def affine_bbox(bboxes, mat):
    bboxes_new = []
    for bbox in bboxes:
        (x0,y0),(x1,y1) = bbox['points']
        cx,cy = (x0 + x1)/2, (y0+y1)/2
        points = np.asarray([
            [cx, y0, 1], [cx,y1,1], [x0,cy,1],[x1,cy,1]
        ])
        points = np.dot(mat, points.transpose()).transpose().astype(np.int32)
        x,y,w,h = cv2.boundingRect(points)
        bbox_new = copy.deepcopy(bbox)
        bbox_new['points'] = ((x,y),(x+w,y+h))
        bboxes_new.append(bbox_new)
    return bboxes_new

def forward(image_path, xml_path, N=3,degree=5):
    rets = []
    image = cv2.imread(image_path,1)
    H,W,_ = image.shape
    bboxes = xmlread(xml_path)
    cx,cy = W/2, H/2
    for _ in range(N):
        deg = random.choice([k for k in range(-degree, degree,1)])
        mat = cv2.getRotationMatrix2D((cx,cy), deg, 1.0)
        image_rot = cv2.warpAffine(image,mat,(W,H), flags = cv2.INTER_AREA,borderMode=cv2.BORDER_REPLICATE)
        bboxes_rot = affine_bbox(bboxes,mat)
        h_rot, w_rot,_ = image_rot.shape
        valid_trans = True #avoid bugs
        for bb in bboxes_rot:
            (x0,y0),(x1,y1) = bb['points']
            x0,x1 = np.clip([x0,x1], 1, W-1)
            y0,y1 = np.clip([y0,y1], 1, H-1)
            if x0 >= x1 or y0 >= y1:
                valid_trans = False
                break
            bb['points'] = (x0,y0),(x1,y1)
        if valid_trans:
            rets.append((image_rot, bboxes_rot))
    return rets





if __name__ == "__main__":
    rets = forward("../DEL/2.jpg","../DEL/2.xml",N=15,degree=45)
    for img, bboxes in rets:
        for bbox in bboxes:
            (x0,y0),(x1,y1) = bbox['points']
            cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),3)
        img = cv2.resize(img,(img.shape[1]//2, img.shape[0]//2))
        cv2.imshow("vis",img)
        cv2.waitKey(-1)


