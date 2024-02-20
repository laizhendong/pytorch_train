import numpy as np 
import cv2
import math
import random
from imgops.anydraw.xml_ops import xmlread,xmlwrite
from numpy.core.fromnumeric import shape
from collections import defaultdict

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def load_image(path,img_size, flag_augment = True):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not flag_augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

def load_anno(path_anno,HW,class_dict):
    H,W = HW
    objects = xmlread(path_anno)
    bboxes = []
    for obj in objects:
        (x0,y0),(x1,y1) = obj['points']
        if obj['name'] not in class_dict.keys():
            class_dict[obj['name']] = len(class_dict.keys())
        class_id = class_dict[obj['name']]
        level = obj['level']
        x0,y0 = x0 / W, y0 / H
        x1,y1 = x1 / W, y1 / H
        cx, cy = (x0 + x1) / 2, (y0 + y1) /2
        w,h = (x1 - x0), (y1 - y0)
        bboxes.append([class_id,cx,cy,w,h,level])
    return np.array(bboxes), class_dict

# labels_all: (classname,cx,cy,w,h,level)
def forward(paths_image, paths_anno, img_size = None, mosaic_border = None,degrees=10,translate=0.1,scale=0.2,shear=10,perspective=0):
    # loads images in a mosaic
    assert len(paths_image) == 4 and len(paths_anno) == 4, "four images/xmls required for mosaic"
    
    if img_size is None:
        img = cv2.imread(paths_image[0],0)
        img_size = max(img.shape[0:2])
        
    if mosaic_border is None:
        mosaic_border = [-img_size//2, -img_size//2]
        
    class_dict = defaultdict(int)

    labels4 = []
    s = img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y
    for i, (path_img, path_anno) in enumerate(zip(paths_image, paths_anno)):
        # Load image
        img, (h0,w0), (h, w) = load_image(path_img, img_size)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x, class_dict = load_anno(path_anno,(h0,w0),class_dict)
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh

        if 0:
            vis = img4.copy()
            for (c, x0, y0, x1, y1, level) in labels:
                cv2.rectangle(vis,(int(x0),int(y0)),(int(x1),int(y1)),(255,0,0),3)
            cv2.imwrite("vis.jpg",vis)
        if labels.size > 0:
            labels4.append(labels)

    #Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_perspective

        img4, labels4 = random_perspective(img4, labels4,
                                        degrees=degrees,
                                        translate=translate,
                                        scale=scale,
                                        shear=shear,
                                        perspective=perspective,
                                        border=mosaic_border)  # border to remove

        cid_dict = defaultdict(str)
        for key in class_dict.keys():
            cid = class_dict[key]
            cid_dict[cid] = key

        bboxes = []
        for one in labels4:
            c,x0,y0,x1,y1,l = one
            c = cid_dict[int(c)]
            info = {"type": "rect",
                "points": [(int(x0),int(y0)), (int(x1), int(y1))],
                "color": "white",
                "name": c,
                "level": l}
            bboxes.append(info)
        return img4, bboxes
    else:
        return None, None


if __name__ == "__main__":
    import glob,os,sys
    paths_image = glob.glob(r"G:\_stain\zz\*.jpg")
    paths_anno = [os.path.splitext(path)[0] + ".xml" for path in paths_image]

    image, bboxes = forward(paths_image, paths_anno, mosaic_border=None)
    jpg = r"F:\__GDisk\dataset\poy_fdy_scratch_mark\_codes\utils\output.jpg"
    cv2.imwrite(jpg,image)
    xml = r"F:\__GDisk\dataset\poy_fdy_scratch_mark\_codes\utils\output.xml"
    xmlwrite(xml,bboxes)
    