#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
from pathlib import Path

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/coco"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.num_classes = 1    
        self.data_num_workers = 1
        self.eval_interval = 1
        self.input_size = (512, 512)
        self.max_epoch = 100
        self.test_size = (512, 512)
        self.print_interval = 100
        self.random_size = (32, 32)
        self.warmup_epochs = 5
        self.warmup_lr = 0.001
