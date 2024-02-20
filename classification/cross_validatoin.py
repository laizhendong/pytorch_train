import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from config.default_config import get_default_config

from symbol.resnet import RESNET

from data.default_dataset import CLASSICAL_CLASSIFIER_BASELINE
from utils.torch_utils import select_device
from train import train_net,setup_logging
from collections import defaultdict

WORK_DIR = os.getcwd()

def split_kfold(data_list, K = 5):
    data_all = []
    for data_file in data_list:
        with open(data_file,'r') as f:
            lines = list(f.readlines())
            lines = list(map(lambda x : x.strip(), lines))
            lines = list(filter(lambda x : len(x) != 0,lines))
            data_all.extend(lines)
    data_all = list(set(data_all))
    class_dict = defaultdict(list)
    for one in data_all:
        #path = one.split(' ')[0]
        labels = ' '.join(one.split(' ')[1:])
        class_dict[labels].append(one.strip())

    for c in class_dict.keys():
        if len(class_dict[c]) < K:
            logging.error(f"sample of class {c} is less than {K}")
            return []

    kfolds = [[] for _ in range(K)]
    for c in class_dict.keys():
        k = 0
        for n in range(len(class_dict[c])):
            kfolds[k].append( class_dict[c][n] )
            k += 1
            if k >= K:
                k = 0
    return kfolds

def write_data_list(train_list, test_list):
    with open('train.txt','w') as f:
        f.write('\n'.join(train_list))
    with open('test.txt','w') as f:
        f.write('\n'.join(test_list))
    return

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_file", help="yaml configuration file")
    ap.add_argument("-K",default=5,help = "folder count")
    args = ap.parse_args()

    cfg = get_default_config()
    cfg.merge_from_file(args.yaml_file)
    cfg.freeze()

    setup_logging(os.path.join(WORK_DIR, cfg.SOLVER.OUTPUT_FOLDER))

    logging.info("==================CV KFold Start============================")
    device = select_device(cfg.SOLVER.DEVICE)
    #logging.info(f'Using device {cfg.SOLVER.DEVICE}')

    kfolds = split_kfold( [cfg.DATA.TRAIN_LIST,cfg.DATA.TEST_LIST], K = args.K )

    val_scores = []
    for k_for_test in range(args.K):
        test_list = kfolds[k_for_test]
        train_list = []
        for k in range(args.K):
            if k == k_for_test:
                continue
            train_list.extend(kfolds[k])
        write_data_list(train_list, test_list)
        if cfg.DATA.NAME == "classical_classifier":
            dataset_train = CLASSICAL_CLASSIFIER_BASELINE(list_file="train.txt", train_flag=True, cfg=cfg)
            dataset_test = CLASSICAL_CLASSIFIER_BASELINE(list_file='test.txt', train_flag=False, cfg=cfg)
        else:
            logging.error(f"unk dataset {cfg.DATA.NAME}")
            sys.exit(0)

        class_num = dataset_train.class_num()

        net = RESNET(backbone_name=cfg.MODEL.BACKBONE_NAME,
                     pretrained_backbone=cfg.MODEL.USE_IMAGENET_BACKBONE,
                     input_channels=dataset_train.input_channels(),
                     num_classes=dataset_train.class_num())
        logging.info(f'Network:\n'
                     f'\t{class_num} output channels (classes)\n')

        if cfg.MODEL.WEIGHTS != "":
            weights = torch.load(cfg.MODEL.WEIGHTS)
            net.load_state_dict(weights, strict=False)

        net.to(device=device)

        try:
            val_score = train_net(cfg, net, dataset_train, dataset_test, device)
            val_scores.append(val_score)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
        logging.info(f"CV KFold {k_for_test + 1}/{args.K}  val score {val_score}")
    logging.info("==================CV KFold Summary============================")
    logging.info(f"{args.K}Kolds val score mean {np.mean(val_scores)} std {np.std(val_scores)}")


