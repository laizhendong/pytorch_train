#import argparse
#import logging
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import shutil

from symbol import resnet
from data import default_dataset
from collections import Counter
from config.default_config import get_default_config
from symbol.repvgg import get_RepVGG_func_by_name, func_dict
from utils.torch_utils import get_num_classes

def predict_img(net,image_data,device,transforms=None):
    H, W, _ = image_data.shape
    if not transforms is None:
        for augment in transforms:
            image_data = augment.forward(image_data)

    net.eval()

    image_data = np.transpose(image_data,(2,0,1))
    image_data = torch.from_numpy(image_data).unsqueeze(0)
    image_data = image_data.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(image_data)
        probs = F.softmax(output, dim=1)
        probs = probs.squeeze(0)
        probs = probs.squeeze().cpu().numpy()

    labels_data = np.argmax(probs,axis=0)
    probs_data = np.max(probs,axis=0)

    return probs_data, labels_data



def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        #logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files




if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_file",help="yaml configuration file")
    ap.add_argument("--model_path",help="yaml configuration file")
    ap.add_argument("--src_dir",help="yaml configuration file")
    ap.add_argument("--dst_dir",help="yaml configuration file")
    args = ap.parse_args()

    
    WORK_DIR = os.getcwd()

    ymal_file = osp.join(WORK_DIR, args.yaml_file)
    model_path = osp.join(WORK_DIR, args.model_path)
    src_dir = osp.join(WORK_DIR, args.src_dir)
    dst_dir = osp.join(WORK_DIR, args.dst_dir)
    

    cfg = get_default_config()
    cfg.merge_from_file(ymal_file)
    cfg.freeze()
    os.makedirs(dst_dir, exist_ok=True)


    num_classes =  max(cfg.ONNX.NUM_CLASSES, get_num_classes(cfg.DATA.TRAIN_LIST))
    classes = num_classes
    if len(classes) < 1:
        classes = [f"{k}" for k in range(num_classes)]
    elif len(classes) != num_classes:
        print("[ERROR] CLASSES {classes} different from {num_classes}")
        exit(0)
    # build net 
    backbone_name =  cfg.MODEL.BACKBONE_NAME
    if backbone_name.startswith('resnet'):
        net = resnet.RESNET(backbone_name = cfg.MODEL.BACKBONE_NAME, input_channels = 3, num_classes=num_classes, pretrained_backbone=False)
        net.load_state_dict(torch.load(model_path), strict=False)
    elif backbone_name.startswith('RepVGG'):
        repvgg_model_list = list(func_dict.keys())
        if not backbone_name in repvgg_model_list :
            raise Exception("wrong name with repvgg , should be in {}, but get {}".format(repvgg_model_list, backbone_name))

        repvgg_build_func = get_RepVGG_func_by_name(backbone_name)
        net = repvgg_build_func(deploy=True, num_classes=num_classes)
        net.load_state_dict(torch.load(model_path), strict=True)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)

    transforms = default_dataset.ClassificationDataset(cfg=cfg,list_file=None,train_flag=False).augments

    preds = []
    for fn in tqdm(os.listdir(src_dir)):

        ext = os.path.splitext(fn)[-1]
        if ext not in {'.jpg', '.bmp', '.png','.jpeg'}:
            continue
        fn = os.path.join(src_dir,fn)
        image_data = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), 1)

        probs_data, labels_data = predict_img(net=net,image_data=image_data,transforms=transforms,device=device)

        labels_name = classes[int(labels_data)]
        preds.append(labels_name)

        outpath = os.path.join(dst_dir, f"{labels_name}", os.path.basename(fn))
        os.makedirs(os.path.dirname(outpath),exist_ok=True)
        shutil.copy(fn, os.path.dirname(outpath))
        #cv2.imwrite(outpath, image_data)

    preds_stat = Counter(preds)
    for name in preds_stat:
        print(f" {preds_stat[name]} : {name} ")


