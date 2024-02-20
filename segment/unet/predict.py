#import argparse
#import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import cv2
#from torchvision import transforms

from unet import UNet
#from utils.data_vis import plot_img_and_mask
from utils.dataset import JiangsiDataset

import copy



CLASS_NUM = 2
MODEL_PATH = "output/CP_final.pth"
PROB_TH = 0.5

INPUT_FOLDER = "G:/_jiangsi/input-train/"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER,"__OUTPUT__")

def predict_img(net,image_data,device, prob_th,transforms=None):
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
    zeros_data = np.zeros_like(probs_data)
    probs_data = np.where(probs_data > prob_th, probs_data, zeros_data)
    if not transforms is None:
        probs_data = cv2.resize(probs_data,(W,H), interpolation=cv2.INTER_NEAREST)
        labels_data = cv2.resize(labels_data, (W, H), interpolation=cv2.INTER_NEAREST)

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

    os.makedirs(OUTPUT_FOLDER,exist_ok=True)


    net = UNet(n_channels=3, n_classes=CLASS_NUM,bilinear=False)

    net.load_state_dict(torch.load(MODEL_PATH))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)

    transforms = JiangsiDataset(list_file=None,train_flag=False).augments


    for i, fn in enumerate(os.listdir(INPUT_FOLDER)):

        ext = os.path.splitext(fn)[-1]
        if ext not in {'.jpg', '.bmp', '.png'}:
            continue
        fn = os.path.join(INPUT_FOLDER,fn)
        image_data = cv2.imread(fn,1)

        probs_data, labels_data = predict_img(net=net,image_data=image_data,transforms=transforms,device=device, prob_th=PROB_TH)

        #to binary
        labels_data = (labels_data > 0).astype(np.int32)
        labels_data = np.expand_dims(labels_data,-1)
        probs_data = np.expand_dims(probs_data, -1)


        labels_data = np.tile(labels_data,(1,1,image_data.shape[-1]))
        image_mask_data = copy.deepcopy(image_data)
        image_mask_data[:,:,2] = (image_data[:,:,2] * 0.5 + labels_data[:,:,2] * 255 * 0.5).astype(np.uint8)

        blend_data = np.where(labels_data, image_mask_data, image_data)

        outpath = os.path.join(OUTPUT_FOLDER, os.path.basename(fn))
        cv2.imwrite(outpath, blend_data)


