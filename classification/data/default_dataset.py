
import numpy as np
import torch
from torch.utils.data import Dataset
import utils.augments as augops
import cv2,os
from tqdm import tqdm
from collections import defaultdict, Counter


class BasicDataset(Dataset):
    def __init__(self,cfg,list_file, train_flag=True):
        self.items = []
        self.total_each_label = defaultdict(int)
        self.class_present_count = defaultdict(int)
        if list_file:
            with open(list_file, 'r') as f:
                for line in tqdm(f, desc="loading list..."):
                    try:
                        path, label = [x.strip() for x in line.strip().split(' ')]
                        if len(cfg.DATA.SRC_DIR) > 0:
                            path =os.path.join(cfg.DATA.SRC_DIR, path)
                    except Exception as e:
                        print(line)
                        print(e)
                        continue
                    label = int(label)
                    self.total_each_label[label] += 1
                    self.items.append((path, label))
        if train_flag:
            self.augments = [
                augops.RandomBlur(cfg),
                augops.RandomRotation(cfg),
                augops.RandomLight(cfg),
                augops.RandomHFlip(cfg),
                augops.RandomVFlip(cfg),
                augops.ShuffleColor(cfg),
                augops.Resize(cfg),
                augops.Crop(cfg,random_crop=True),
                augops.ToTensor(cfg.DATA.MEAN, cfg.DATA.STD)
            ]


        else:
            self.augments = [
                augops.Resize(cfg),
                augops.Crop(cfg,random_crop=False),
                augops.ToTensor(cfg.DATA.MEAN, cfg.DATA.STD)
            ]


    def __len__(self):
        return len(self.items)

    def get_index_wrt_label(self):
        data = defaultdict(list)
        for index,one in enumerate(self.items):
            data[one[-1]].append(index)
        return data

    def class_size(self):
        data = []
        for k in range(self.class_num()):
            data.append(self.total_each_label[k])
        return data

    def input_channels(self):
        pass

    def class_num(self):
        return len(self.total_each_label.keys())


    def __getitem__(self, item):
        return


class ClassificationDataset(BasicDataset):
    def __init__(self, **kwargs):
        super(ClassificationDataset,self).__init__(**kwargs)
    def __getitem__(self, item):
        image, label = self.items[item]
        #image_data = cv2.imread(image,1)
        image_data = cv2.imdecode(np.fromfile(image,dtype=np.uint8),1) #support chinese

        for augment in self.augments:
            image_data, _ = augment.forward((image_data, label))


        image_data = np.transpose(image_data,(2,0,1))


        return {
            'image': torch.from_numpy(image_data).type(torch.FloatTensor),
            'label': torch.from_numpy(np.asarray([label])).type(torch.FloatTensor)
        }

class DualClassificationDataset(BasicDataset):
    def __init__(self, merge_mode, **kwargs):
        super(DualClassificationDataset, self).__init__(**kwargs)
        self.merge_mode = merge_mode
    def __getitem__(self, item):
        image, label = self.items[item]
        image_data = cv2.imread(image, 1)
        dual_image_data = cv2.imread(os.path.splitext(image)[0] + ".png",0)

        for augment in self.augments:
            image_data, mask_data = augment.forward((image_data, label))

        H, W, C = image_data.shape
        dual_image_data = cv2.resize(dual_image_data, (W,H), cv2.INTER_NEAREST)

        image_data = np.transpose(image_data, (2, 0, 1))
        dual_image_data = np.expand_dims(dual_image_data,axis=0).astype(np.float32)

        if self.merge_mode == "concat":
            image_data = np.concatenate((image_data, dual_image_data//255.0),axis=0)
        else:
            image_data = image_data + dual_image_data / 255.0

        return {
            'image': torch.from_numpy(image_data).type(torch.FloatTensor),
            'label': torch.from_numpy(np.asarray([label])).type(torch.FloatTensor)
        }


class CLASSICAL_CLASSIFIER_BASELINE(ClassificationDataset):
    def __init__(self,**kwargs):
        super(CLASSICAL_CLASSIFIER_BASELINE, self).__init__(**kwargs)
        return
    def input_channels(self):
        return 3

