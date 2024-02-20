# encoding=utf-8
import os,shutil
from datetime import datetime

class CONFIG:
    def __init__(self):
        self.datalist = {
            "train": "G:\\_jiangsi\\trainval\\trainval.txt",
            "test": "G:\\_jiangsi\\trainval\\trainval.txt"
        }
        self.network = {
            "name":"unet", #只能是unet
            "pretrained":None, #预训练模型的路径
            "bilinear":False #TRT5不支持UpSampling()，这里只能设置成False
        }


        self.data_params = {
            "mean":[.485, .456, .406], #均值
            "std": [.225, .225, .225], #方差
            "auto_class_weight":True, #是否自动计算类别权重，推荐True
            'batch_size': 6 #batch size
        }
        self.epochs = 100 #epoch
        self.threads = 0 #数据加载的线程个数
        self.augment_params = { #一些增广策略，注释即可禁用对应的策略
            #具体含义可以参考utils/augments.py
            #"shear":(0.05,0.05), #shear变换，参见RandomShear
            #"blur":3, #模糊，参见RandomBlur
            #"gamma":0.4, #gamma亮化，参见RandomLight
            #"hflip":True, #水平翻转 参见RandomHFlip
            #'vflip':True, #竖直翻转 参见RandomVFlip
            #"color":False, #颜色变换 参见RandomColor
            "resize":(224,224,1) #w,h,inter #这一项必须设置，表示网络输入尺寸，其中最后一项表示采样方法，具体参见ToTensor
        }

        self.lr_params = {
            "base_lr": 0.001, #初始学习率
            "policy": "cosine" #学习率调整策略，目前只支持cosine和plateau
        }

        self.clip_value = 0.1 #梯度裁剪，设置成负数可以禁用

        self.valid_freq = 10 #验证周期
        
        output_folder = "output/{}_{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M'), self.network['name'])
        output_folder = os.path.join(os.path.dirname(__file__),output_folder)
        self.log_dir = os.path.join(output_folder, "log") #日志输出路径
        self.saved_dir = os.path.join(output_folder, "weights") #模型保存路径

    def build_output(self):
        os.makedirs(self.log_dir,exist_ok=True)
        os.makedirs(self.saved_dir,exist_ok=True)

        output_folder = os.path.dirname(self.log_dir)
        if not os.path.exists(os.path.join(output_folder,"config.py")):
            shutil.copy(__file__,output_folder)
        return


CFG = CONFIG()