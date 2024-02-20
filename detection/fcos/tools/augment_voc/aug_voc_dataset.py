# encoding = utf-8
import os,time
import cv2,shutil
import json
import warnings
from utils import zoom,rotation,paste_targets, mosaic_image
from imgops.anydraw.xml_ops import  xmlwrite
import random
import logging
from tqdm import tqdm
from utils import setup_logging

with open("config.json","r", encoding="utf-8") as f:
    CFG = json.load(f)

VOC_ROOT = CFG["aug"]["input_dir"]
VOC_AUG_ROOT = CFG["aug"]["output_dir"]
MIN_SIZE = CFG["aug"]["min_size"]

AUG_ZOOM = CFG['aug']['zoom']
AUG_PASTE_TARGET = CFG['aug']["paste_target"]
AUG_ROT = CFG['aug']['rotation']
AUG_MOSAIC = CFG['aug']['mosaic']
if MIN_SIZE > 0:
    warnings.warn(f"min_size is {MIN_SIZE} and small bbox will be updated")

voc_aug_images = os.path.join(VOC_AUG_ROOT, "JPEGImages")
voc_aug_anno = os.path.join(VOC_AUG_ROOT, "Annotations")
os.makedirs(voc_aug_images,exist_ok=True)
os.makedirs(voc_aug_anno,exist_ok=True)
###################################################

def get_xml_head(image):
    H,W,_ = image.shape
    return {
        "width":W,"height":H,"depth":3
    }

#尝试扩展小尺寸的标记
def check_min_size(W,H,bbox):
    (x0, y0), (x1, y1) = bbox['points']
    if MIN_SIZE > 0:
        w, h = x1 - x0, y1 - y0
        if w < MIN_SIZE:
            cx = (x1 + x0) // 2
            x0 = max([0, cx - MIN_SIZE // 2])
            x1 = min([cx + MIN_SIZE // 2, W - 1])
        if h < MIN_SIZE:
            cy = (y1 + y0) // 2
            y0 = max([0, cy - MIN_SIZE // 2])
            y1 = min([cy + MIN_SIZE // 2, H - 1])
        bbox['points'] = ((x0, y0), (x1, y1))
    return bbox


def augment_dataset(splits = "trainval"):
    #只对指定集合做增广
    path = os.path.join(VOC_ROOT,"ImageSets","Main",f"{splits}.txt")
    with open(path,'r') as f:
        keys = f.readlines()
        keys = map(lambda x: x.strip(), keys)
        keys = filter(lambda x: x != "", keys)

    data_all = []
    for key in keys:
        try:
            xml = os.path.join(VOC_ROOT,"Annotations", f"{key}.xml")
            image = os.path.join(VOC_ROOT,"JPEGImages", f"{key}.jpg")
            data_all.append((image,xml))
        except Exception as e:
            warnings.warn(f">>ERROR {key}")
            warnings.warn(f"{e}")
            continue


    augments_keys = []
    if AUG_ZOOM > 0:
        for (image,xml) in tqdm(data_all,desc="zoom"):
            zooms = zoom.forward(image,xml,total=AUG_ZOOM)
            for image_aug, bboxes_aug in zooms:
                H,W = image_aug.shape[0],image_aug.shape[1]
                #检查bbox尺寸
                bboxes_aug_valid = []
                for bbox in bboxes_aug:
                    bbox = check_min_size(W,H,bbox)
                    if bbox is None:
                        continue
                    bboxes_aug_valid.append(bbox)
                bboxes_aug = bboxes_aug_valid
                if bboxes_aug == []:
                    continue

                key = f"zoom_{len(augments_keys)}"
                augments_keys.append(key)
                cv2.imwrite(os.path.join(voc_aug_images,key + ".jpg"),image_aug)
                xmlwrite(os.path.join(voc_aug_anno, key + ".xml"),bboxes_aug, head=get_xml_head(image_aug))

    if AUG_ROT > 0:
        for (image, xml) in tqdm(data_all,desc="rot"):
            rots = rotation.forward(image, xml, N = AUG_ROT)
            for image_aug, bboxes_aug in rots:
                H, W = image_aug.shape[0], image_aug.shape[1]
                # 检查bbox尺寸
                bboxes_aug_valid = []
                for bbox in bboxes_aug:
                    bbox = check_min_size(W, H, bbox)
                    if bbox is None:
                        continue
                    bboxes_aug_valid.append(bbox)
                bboxes_aug = bboxes_aug_valid
                if bboxes_aug == []:
                    continue

                key = f"rot_{len(augments_keys)}"
                augments_keys.append(key)
                cv2.imwrite(os.path.join(voc_aug_images, key + ".jpg"), image_aug)
                xmlwrite(os.path.join(voc_aug_anno, key + ".xml"), bboxes_aug, head=get_xml_head(image_aug))

    if 0:
        random.shuffle(data_all)
        N = len(data_all)//4
        for index in range(0,N,4):
            images = [x[0] for x in data_all[index:index+4]]
            xmls = [x[1] for x in data_all[index:index + 4]]
            image_aug, bboxes_aug = paste_image.forward(images,xmls)

            # 检查bbox尺寸
            bboxes_aug_valid = []
            H,W = image_aug.shape[0],image_aug.shape[1]
            for bbox in bboxes_aug:
                bbox = check_min_size(W, H, bbox)
                if bbox is None:
                    continue
                bboxes_aug_valid.append(bbox)
            bboxes_aug = bboxes_aug_valid
            if bboxes_aug == []:
                continue

            key = f"combine_{len(augments_keys)}"
            augments_keys.append(key)
            cv2.imwrite(os.path.join(voc_aug_images,key + ".jpg"),image_aug)
            xmlwrite(os.path.join(voc_aug_anno, key + ".xml"),bboxes_aug, head=get_xml_head(image_aug))



    if AUG_MOSAIC > 0:
        indices = [k for k in range(len(data_all))]
        for index in tqdm(range(AUG_MOSAIC*len(data_all)),desc="mosaic"):
            indices_selected = random.choices(indices,k=3)
            indices_selected.append(index%len(data_all))
            images = [data_all[k][0] for k in indices_selected]
            xmls = [data_all[k][1] for k in indices_selected]
            image_aug, bboxes_aug = mosaic_image.forward(images,xmls)

            # 检查bbox尺寸
            bboxes_aug_valid = []
            H,W = image_aug.shape[0],image_aug.shape[1]
            for bbox in bboxes_aug:
                bbox = check_min_size(W, H, bbox)
                if bbox is None:
                    continue
                bboxes_aug_valid.append(bbox)
            bboxes_aug = bboxes_aug_valid
            if bboxes_aug == []:
                continue

            key = f"mosaic_{len(augments_keys)}"
            augments_keys.append(key)
            cv2.imwrite(os.path.join(voc_aug_images,key + ".jpg"),image_aug)
            xmlwrite(os.path.join(voc_aug_anno, key + ".xml"),bboxes_aug, head=get_xml_head(image_aug))

    if AUG_PASTE_TARGET:
        random.shuffle(data_all)
        N = len(data_all)//2
        for index in range(0,N,2):
            images = [x[0] for x in data_all[index:index+2]]
            xmls = [x[1] for x in data_all[index:index+2]]
            image_aug, bboxes_aug = paste_targets.forward(images,xmls)


            # 检查bbox尺寸
            bboxes_aug_valid = []
            H,W = image_aug.shape[0],image_aug.shape[1]
            for bbox in bboxes_aug:
                bbox = check_min_size(W, H, bbox)
                if bbox is None:
                    continue
                bboxes_aug_valid.append(bbox)
            bboxes_aug = bboxes_aug_valid
            if bboxes_aug == []:
                continue

            key = f"paste_{len(augments_keys)}"
            augments_keys.append(key)
            cv2.imwrite(os.path.join(voc_aug_images,key + ".jpg"),image_aug)
            xmlwrite(os.path.join(voc_aug_anno, key + ".xml"),bboxes_aug, head=get_xml_head(image_aug))

    outdir = os.path.join(VOC_AUG_ROOT,"ImageSets","Main")
    os.makedirs(outdir,exist_ok=True)
    with open(os.path.join(outdir,"train.txt"),"w") as f:
        f.write('\n'.join(augments_keys))
    with open(os.path.join(outdir, "trainval.txt"), "w") as f:
        f.write('\n'.join(augments_keys))

if __name__=="__main__":
    setup_logging.setup()
    running_info = time.asctime(time.localtime(time.time()))
    logging.info(f"------------Start {running_info}------------------------")
    logging.info(f"{CFG}")
    augment_dataset()

