
import numpy as np
import torch
from torch.utils.data import Dataset
import utils.augments as augops
import cv2,os
from tqdm import tqdm
from config import CFG
from collections import defaultdict, Counter


class BasicDataset(Dataset):
    def __init__(self, list_file, train_flag=True):
        self.items = []
        self.total_each_label = defaultdict(int)
        self.class_present_count = defaultdict(int)
        if list_file:
            with open(list_file, 'r') as f:
                for line in tqdm(f, desc="loading list..."):
                    try:
                        image, mask = [x.strip() for x in line.strip().split(' ')]
                    except Exception as e:
                        print(line)
                        print(e)
                        continue
                    if os.path.splitext(mask)[-1] == '.rle':
                        image_data = cv2.imread(image, 1)
                        labels_one = self.counter_rle(image_data, mask)
                    else:
                        labels_one = cv2.imread(mask, 0).flatten().tolist()
                    label_counter_one = Counter(labels_one)

                    for label in label_counter_one.keys():
                        self.total_each_label[label] += label_counter_one[label]
                        self.class_present_count[label] += 1
                    self.items.append((image, mask))
        if train_flag:
            self.augments = [
                augops.RandomBlur(),
                augops.RandomShear(),
                augops.RandomLight(),
                augops.RandomHFlip(),
                augops.RandomVFlip(),
                augops.Resize(),
                augops.ToTensor(CFG.data_params['mean'], CFG.data_params['std'])
            ]


        else:
            self.augments = [
                augops.Resize(),
                augops.ToTensor(CFG.data_params['mean'], CFG.data_params['std'])
            ]

    def counter_rle(self, image_data, rlepath, target_ids = None):
        if target_ids is None:
            target_ids = {3:1,4:1}
        H,W,C = image_data.shape
        pts = np.zeros((W*H,),dtype=np.uint8)
        with open(rlepath,'r') as f:
            for line in f:
                data = [int(x) for x in line.strip().split(',')]
                target_id = data[0]
                if target_id not in target_ids.keys():
                    continue
                for start,end in zip(data[1::2], data[2::2]):
                    pts[start:end+1] = target_ids[target_id]
        return pts

    def load_rle(self, image_data, rlepath, target_ids = None):
        if target_ids is None:
            target_ids = {3:1,4:1}
        H,W,C = image_data.shape
        mask = np.zeros((H*W,),dtype=np.uint8)
        pts = {}
        with open(rlepath,'r') as f:
            for line in f:
                data = [int(x) for x in line.strip().split(',')]
                target_id = data[0]
                if target_id not in target_ids.keys():
                    continue
                pts[target_id] = data[1:]

        for label in pts.keys():
            xy = pts[label]
            relabel = target_ids[label]
            for start, end in zip(xy[0::2], xy[1::2]):
                mask[start:end+1] = relabel
        return np.reshape(mask,(H,W))

    def __len__(self):
        return len(self.items)

    def class_size(self):
        data = []
        for k in range(self.class_num()):
            data.append(self.total_each_label[k])
        return data

    def class_present(self):
        data = []
        for k in range(self.class_num()):
            data.append(self.class_present_count[k])
        return data

    def class_num(self):
        return len(self.total_each_label.keys())


    def __getitem__(self, item):
        image, mask = self.items[item]
        image_data = cv2.imread(image,1)
        if os.path.splitext(mask)[-1] == ".rle":
            mask_data = self.load_rle(image_data,mask)
        else:
            mask_data = cv2.imread(mask, 0)

        for augment in self.augments:
            image_data, mask_data = augment.forward((image_data, mask_data))

        if len(mask_data.shape) == 2:
            mask_data = np.expand_dims(mask_data,axis=2)

        image_data = np.transpose(image_data,(2,0,1))
        mask_data = np.transpose(mask_data, (2, 0, 1))

        return {
            'image': torch.from_numpy(image_data).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask_data).type(torch.FloatTensor)
        }


class JiangsiDataset(BasicDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
