
import cv2,os
from config import CFG
import numpy as np
import random




class RandomBlur(object):
    def __init__(self):
        super(RandomBlur, self).__init__()
        self._win_size = 0
        self._name = "blur"
        self._flag_on = False
        if self._name in CFG.augment_params.keys():
            self._win_size = CFG.augment_params[self._name]
            self._flag_on = True


    def forward(self, x):
        if not self._flag_on or random.uniform(0,1) < 0.5:
            return x
        img = x[0].astype(np.uint8)
        #cv2.imshow("blur-input", img.astype(np.uint8))
        blur_type = random.randint(0,3)
        blur_param = 1 + 2 * random.randint(1,self._win_size//2)

        if blur_type == 0:
            img = cv2.GaussianBlur(img,(blur_param,blur_param),0)
        elif blur_type == 1:
            img = cv2.blur(img,(blur_param,blur_param))
        elif blur_type == 2:
            img = cv2.medianBlur(img,blur_param)
        elif blur_type == 3:
            img = cv2.boxFilter(img,-1,(blur_param,blur_param))
        #cv2.imshow("blur",img.astype(np.uint8))
        #cv2.waitKey(-1)
        return img, x[1]




class Resize(object):
    def __init__(self):
        super(Resize, self).__init__()
        self._width = 0
        self._height = 0
        self._inter = 1
        self._name = "resize"
        self._flag_on = False
        if self._name in CFG.augment_params.keys():
            self._width, self._height, self._inter = CFG.augment_params[self._name]
            self._flag_on = True


    def forward(self, x):
        if not self._flag_on:
            return x
        if isinstance(x,(list,tuple)) and len(x) == 2:
            img = x[0].astype(np.uint8)
            img = cv2.resize(img,(self._width, self._height),interpolation=self._inter)

            label = x[1].astype(np.uint8)
            label = cv2.resize(label, (self._width, self._height), interpolation=cv2.INTER_NEAREST)
            return img, label
        else:
            img = x.astype(np.uint8)
            img = cv2.resize(img, (self._width, self._height), interpolation=self._inter)
            return img

class RandomShear(object):
    def __init__(self):
        super(RandomShear, self).__init__()
        self._dx,self._dy = 0,0
        self._name = "shear"
        self._flag_on = False
        if self._name in CFG.augment_params.keys():
            self._dx = CFG.augment_params[self._name][0]
            self._dy = CFG.augment_params[self._name][1]
            self._flag_on = True

    def forward(self, x):
        if not self._flag_on or random.uniform(0,1) < 0.5:
            return x
        img = x[0].astype(np.uint8)
        label = x[1].astype(np.uint8)
        dx = random.uniform(-1 * self._dx, self._dx)
        dy = random.uniform(-1 * self._dy, self._dy)
        M = np.zeros((2,3))
        M[0,0] = 1
        M[1,1] = 1
        M[0,1] = dx
        M[1,0] = dy
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT)

        label = cv2.warpAffine(label, M, (img.shape[1], img.shape[0]), flags = cv2.INTER_NEAREST, borderMode = cv2.BORDER_REFLECT)
        return img, label


class RandomLight(object):
    def __init__(self):
        super(RandomLight, self).__init__()
        self._name = "gamma"
        self._gamma_delta = -1
        self._flag_on = False
        if self._name in CFG.augment_params.keys():
            self._gamma_delta = CFG.augment_params[self._name]
            self._flag_on = True



    def forward(self, x):
        if not self._flag_on or random.uniform(0,1) < 0.5:
            return x
        img = x[0].astype(np.float32)
        #cv2.imshow("gamma-input", img.astype(np.uint8))
        gamma = random.uniform(1-self._gamma_delta, 1 + self._gamma_delta)
        img = np.clip(np.power( img / 255.0, gamma) * 255,0,255)
        #cv2.imshow("gamma",img.astype(np.uint8))
        #cv2.waitKey(-1)
        return img, x[1]


class RandomHFlip(object):
    def __init__(self):
        super(RandomHFlip, self).__init__()
        self._name = "hflip"
        self._flag_on = False
        if self._name in CFG.augment_params.keys() and CFG.augment_params[self._name]:
            self._flag_on = True

    def forward(self, x):
        if not self._flag_on or random.uniform(0, 1) < 0.5:
            return x
        img = x[0]
        img = cv2.flip(img,1)

        label = x[1]
        label = cv2.flip(label, 1)
        return img, label

class RandomVFlip(object):
    def __init__(self):
        super(RandomVFlip, self).__init__()
        self._name = "vflip"
        self._flag_on = False
        if self._name in CFG.augment_params.keys() and CFG.augment_params[self._name]:
            self._flag_on = True

    def forward(self, x):
        if not self._flag_on or random.uniform(0, 1) < 0.5:
            return x
        img = x[0]
        img = cv2.flip(img,0)

        label = x[1]
        label = cv2.flip(label, 0)
        return img, label


class RandomColor(object):
    def __init__(self):
        super(RandomColor, self).__init__()
        self._name = "color"
        self._flag_on = False
        if self._name in CFG.augment_params.keys() and CFG.augment_params[self._name]:
            self._flag_on = True

    def forward(self, x):
        if not self._flag_on or random.uniform(0, 1) < 0.5:
            return x
        img = x[0]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = hsv[:,:,0].astype(np.int32)
        hue =  ((hue + random.randint(0,360)) % 180).astype(np.uint8)
        hsv[:, :, 0] = hue
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img, x[1]


class ToTensor(object):
    def __init__(self, mean,std):
        super(ToTensor, self).__init__()
        self.mean_ = np.asarray(mean).reshape((1,1,-1))
        self.std_  = np.asarray(std).reshape((1,1,-1))
        return

    def forward(self, x):
        if isinstance(x, (list, tuple)) and len(x) == 2:
            img = (x[0] / 256.0 - self.mean_) / self.std_
            return img,x[1]
        else:
            img = (x / 256.0 - self.mean_) / self.std_
            return img



