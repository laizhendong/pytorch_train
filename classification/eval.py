import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cv2
import numpy as np

from tqdm import tqdm
import argparse

from utils.metrics import MetricsAll
from data.robbin_break_topborder import CLASSICAL_CLASSIFIER_BASELINE
from config.default_config import get_default_config
from utils.torch_utils import select_device
from symbol import resnet
import torchvision
from symbol.repvgg import get_RepVGG_func_by_name, func_dict, repvgg_model_convert
import time

def print_info(info, color=None):
    if color == 'red':
        print("\033[91m{}\033[0m".format(info))
    elif color == 'yellow':
        print("\033[93m{}\033[0m".format(info))
    elif color == 'blue':
        print("\033[94m{}\033[0m".format(info))
    else:
        print(info)

def eval_net(net, loader, device, class_num, vis=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    MetricsObject = MetricsAll(class_num, name="Validset")

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['label']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_labels = true_masks.to(device=device, dtype=torch.long)
            with torch.no_grad():
                label_pred = net(imgs)


            tot += F.cross_entropy(label_pred, torch.squeeze(true_labels).long()).item()

            if vis:
                for idx in range(imgs.shape[0]):
                
                    im_tensor = imgs[idx, :, :, :]
                    im_numpy = im_tensor.cpu().numpy()
                    im_numpy = np.transpose(im_numpy, (1, 2, 0))

                    gt_label = int(true_labels.cpu()[idx][0].detach())
                    pred_label = int(torch.argmax(label_pred.cpu()[idx]).detach())

                    save_dir = 'label_' + str(gt_label) + '/pred_' + str(pred_label)
                    save_dir = osp.join('./inference', save_dir)
                    if not osp.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    cv2.imwrite(osp.join(save_dir, str(k)+'.jpg'), im_numpy)


            MetricsObject.update(true_labels.cpu(), label_pred.cpu())

            pbar.update()

    net.train()
    return tot / n_val, MetricsObject.get("f1score"), MetricsObject.get("accuracy")



if __name__ == "__main__":

    import os
    import os.path as osp

    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_file",help="yaml configuration file")
    ap.add_argument("--model_path",help="model path to eval")
    args = ap.parse_args()


    work_dir = os.getcwd()
    print_info("------- eval model -------------- \n", 'red')
    print_info(" Current workspace is {} ".format(work_dir))
    print_info("\n-------  -------------- \n", 'red')
    yaml_fullpath = osp.join(work_dir, args.yaml_file).strip()
    model_fullpath = osp.join(work_dir, args.model_path).strip()

    # check param
    if not os.path.exists(yaml_fullpath):
        raise Exception("no such config file {}".format(yaml_fullpath))
    if not os.path.exists(model_fullpath):
        raise Exception("no such model file {}".format(model_fullpath))

    # read config
    cfg = get_default_config()
    cfg.merge_from_file(yaml_fullpath)
    cfg.freeze()

    # construct the dataset 
    dataset_test = CLASSICAL_CLASSIFIER_BASELINE(list_file=cfg.DATA.TEST_LIST, train_flag=False, cfg=cfg)
    val_loader = DataLoader(dataset_test,batch_size=cfg.SOLVER.BATCH_SIZE,
                        shuffle=False, num_workers=16, pin_memory=True, drop_last=False)

    # build net
    backbone_name =  cfg.MODEL.BACKBONE_NAME
    if backbone_name.startswith('resnet'):
        net = resnet.RESNET(backbone_name = cfg.MODEL.BACKBONE_NAME, input_channels = 3, num_classes=dataset_test.class_num(), pretrained_backbone=False)
    elif backbone_name.startswith('RepVGG'):
        repvgg_model_list = list(func_dict.keys())
        if not backbone_name in repvgg_model_list :
            raise Exception("wrong name with repvgg , should be in {}, but get {}".format(repvgg_model_list, backbone_name))

        repvgg_build_func = get_RepVGG_func_by_name(backbone_name)
        net = repvgg_build_func(deploy=False, num_classes=dataset_test.class_num(), with_softmax=True) # dataset_test.class_num()

    device = select_device(cfg.SOLVER.DEVICE)

    net.load_state_dict(torch.load(model_fullpath))

    net = net.to(device=device)

    # eval origin net
    if backbone_name.startswith('RepVGG'):
        print_info("repvgg origin deploy eval:", "yellow")

    start_time = time.time()
    val_info = eval_net(net, val_loader, device, dataset_test.class_num())
    end_time = time.time()
    for (name, val) in val_info[1:]:
        print(val)
    print_info("evalutation using time {:.3f} ms".format(end_time - start_time), "blue")
    
    # eval converted net
    if backbone_name.startswith('RepVGG'):
        converted_model_savepath = model_fullpath.replace(".pth", "_converted.pth")
        deploy_repvgg = repvgg_model_convert(net, save_path=converted_model_savepath)
        print_info("\n\nrepvgg converted deploy eval", "red")
        start_time = time.time()
        val_info = eval_net(deploy_repvgg, val_loader, device, dataset_test.class_num())
        end_time = time.time()

        for (name, val) in val_info[1:]:
            print(val)
        print_info("evalutation using time {:.3f} ms".format(end_time - start_time), "blue")

        print_info("save converted repvgg in {}".format(converted_model_savepath), "red")