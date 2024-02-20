

import cv2
import numpy as np
import copy
from imgops.anydraw.xml_ops import xmlread


def forward(image, xml, total = 3, ratio = 0.1):
    #
    image = cv2.imread(image,1)
    H,W,_ = image.shape

    bboxes = xmlread(xml)

    dw, dh = int(ratio*W), int(ratio * H)


    delta_ws = np.random.choice([k for k in range(0, dw)],total)
    delta_hs = np.random.choice([k for k in range(0, dh)],total)

    rets = []
    for dw, dh in zip(delta_ws,delta_hs):
        image_aug = np.zeros((H + dh,W + dw,3),dtype=np.uint8)
        dx,dy = dw//2, dh//2
        image_aug[dy:dy+H, dx:dx+W,:] = copy.deepcopy(image)
        image_aug = cv2.resize(image_aug,(W,H),0,0,cv2.INTER_LINEAR)
        bboxes_aug = []
        sx, sy = W * 1.0 / (W + dw) , H * 1.0 / (H + dh)
        for bbox in bboxes:
            bbox_aug = copy.deepcopy(bbox)
            (x0,y0),(x1,y1) = bbox_aug['points']
            x0,y0 = int((x0 + dx) * sx), int((y0+dy)*sy)
            x1,y1 = int((x1 + dx) * sx), int((y1+dy)*sy)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            bbox_aug['points'] = ( (x0,y0),(x1,y1) )
            bboxes_aug.append(bbox_aug)
        rets.append((image_aug, bboxes_aug))
    return rets








