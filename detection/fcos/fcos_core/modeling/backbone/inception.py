import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import re
def CBR(inp, oup, prefix = "", **kwargs):
    mod =  nn.Sequential()
    mod.add_module("conv", nn.Conv2d(inp, oup, **kwargs))
    mod.add_module(prefix + "bn", nn.BatchNorm2d(oup))
    mod.add_module(prefix + "relu", nn.ReLU(inplace=True))
    return mod

class Inception3A(nn.Module):
    def __init__(self, inp, prefix=""):
        super(Inception3A, self).__init__()
        self.conv1 = CBR(inp, 96,kernel_size=1,stride=1,padding=0)

        self.conv3 = nn.Sequential()
        self.conv3.add_module("1",CBR(inp, 16,kernel_size=1,stride=1,padding=0))
        self.conv3.add_module("2",CBR(16, 64,kernel_size=3, stride=2,padding=1))


        self.conv5 = nn.Sequential()
        self.conv5.add_module("1", CBR(inp, 16,kernel_size=1,stride=1,padding=0))
        self.conv5.add_module("2",CBR(16, 32,kernel_size=3,stride=1,padding=1))
        self.conv5.add_module("3",CBR(32, 32,kernel_size=3,stride=2,padding=1))

        self.pool1 = nn.MaxPool2d(3,stride=2,padding=0,ceil_mode=True)
        return


    def forward(self, x):
        conv1 = self.conv1(self.pool1(x))
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        outputs = [conv1, conv3, conv5]
        return torch.cat(outputs, 1)



class Inception3B(nn.Module):
    def __init__(self, inp,prefix = ""):
        super(Inception3B, self).__init__()
        self.conv1 = CBR(inp, 96, kernel_size=1,stride=1,padding=0)

        self.conv3 = nn.Sequential()
        self.conv3.add_module("1", CBR(inp, 16,kernel_size=1,stride=1,padding=0))
        self.conv3.add_module("2", CBR(16, 64,kernel_size=3, stride=1,padding=1))


        self.conv5 = nn.Sequential()
        self.conv5.add_module("1", CBR(inp, 16, kernel_size=1,stride=1,padding=0))
        self.conv5.add_module("2",CBR(16, 32, kernel_size=3,stride=1,padding=1))
        self.conv5.add_module("3",CBR(32, 32, kernel_size=3,stride=1,padding=1))


        return

    def forward(self, x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        outputs = [conv1, conv3, conv5]
        return torch.cat(outputs, 1)




class Inception4A(nn.Module):
    def __init__(self, inp,prefix=""):
        super(Inception4A, self).__init__()
        self.conv1 = CBR(inp, 128,kernel_size=1,stride=1,padding=0)

        self.conv3 = nn.Sequential()
        self.conv3.add_module("1", CBR(inp, 32, kernel_size=1,stride=1,padding=0))
        self.conv3.add_module("2", CBR(32, 96, kernel_size=3, stride=2,padding=1))

        self.conv5 = nn.Sequential()
        self.conv5.add_module("1", CBR(inp, 16, kernel_size=1,stride=1,padding=0))
        self.conv5.add_module("2", CBR(16, 32, kernel_size=3,stride=1,padding=1))
        self.conv5.add_module("3", CBR(32, 32, kernel_size=3,stride=2,padding=1))

        self.pool1 = nn.MaxPool2d(3,stride=2,padding=0,ceil_mode=True)
        return


    def forward(self, x):
        conv1 = self.conv1(self.pool1(x))
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        outputs = [conv1, conv3, conv5]
        return torch.cat(outputs, 1)



class Inception4B(nn.Module):
    def __init__(self, inp,prefix=""):
        super(Inception4B, self).__init__()
        self.conv1 = CBR(inp, 128, kernel_size=1,stride=1,padding=0)

        self.conv3 = nn.Sequential()
        self.conv3.add_module("1",CBR(inp, 32, kernel_size=1,stride=1,padding=0))
        self.conv3.add_module("2",CBR(32, 96, kernel_size=3, stride=1,padding=1))

        self.conv5 = nn.Sequential()
        self.conv5.add_module("1",CBR(inp, 16, kernel_size=1,stride=1,padding=0))
        self.conv5.add_module("2",CBR(16, 32, kernel_size=3,stride=1,padding=1))
        self.conv5.add_module("3",CBR(32, 32, kernel_size=3,stride=1,padding=1))


        return


    def forward(self, x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        outputs = [conv1, conv3, conv5]
        return torch.cat(outputs, 1)

class PVANET_LITE(nn.Module):
    def __init__(self,cfg,**kwargs):
        super(PVANET_LITE, self).__init__(**kwargs)
        self.stem = nn.Sequential()
        self.stem.add_module("conv1",CBR(3,32,kernel_size=4,stride=2,padding=1))

        self.body = nn.Sequential()

        self.body.add_module("conv2",CBR(32,48,kernel_size=3,stride=2,padding=1))
        self.body.add_module("conv3",CBR(48,96,kernel_size=3,stride=2,padding=1))

        inc3 = nn.Sequential()
        inc3.add_module("a",Inception3A(96))
        inc3.add_module("b", Inception3B(192))
        inc3.add_module("c", Inception3B(192))
        inc3.add_module("d", Inception3B(192))
        inc3.add_module("e", Inception3B(192))
        self.body.add_module("inc3",inc3)


        inc4 = nn.Sequential()
        inc4.add_module("a", Inception4A(192))
        inc4.add_module("b", Inception4B(256))
        inc4.add_module("c", Inception4B(256))
        inc4.add_module("d", Inception4B(256))
        inc4.add_module("e", Inception4B(256))
        self.body.add_module("inc4", inc4)

        if not (cfg is None):
            self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at >= 1:
            for p in self.stem.parameters():
                p.requires_grad = False
        if freeze_at >= 2:
            for index in range(freeze_at-1):
                for p in self.body[index].parameters():
                    p.requires_grad = False
        return

    def __call__(self, input, **kwargs):
        outputs = []
        x = self.stem(input)
        for stage in self.body:
            x = stage(x)
            outputs.append(x)
        return outputs





    def load_caffe_weight(self,path):
        def _update_param_caffe(params_caffe):
            params = {}
            for one in params_caffe:
                name, values_str = one.split(':')
                values = [float(x.strip()) for x in values_str.split(',')]
                name = name.strip()
                name = re.sub("/",".",name)
                name = re.sub("_", ".", name)
                name = re.sub("running.mean", "running_mean", name)
                name = re.sub("running.var", "running_var", name)
                for inc in ['inc3','inc4']:
                    for stage in ['a','b','c','d','e']:
                        name = re.sub(f"{inc}{stage}",f"{inc}.{stage}",name)
                for layer in ['1','2','3']:
                    name = re.sub(f"{layer}.weight",f"{layer}.conv.weight",name)
                    name = re.sub(f"{layer}.bias", f"{layer}.conv.bias", name)
                params[name] = values
            return params
        with open(path,'r') as f:
            params_caffe = f.readlines()
            params_caffe = list(map(
                lambda x: x.strip().strip(','), params_caffe
            ))
            params_caffe = list(filter(
                lambda x: x != "", params_caffe
            ))
        params_caffe = _update_param_caffe(params_caffe)
        return params_caffe



if __name__== "__main__":
    def _load_pva_weight(model):
        params_caffe = model.load_caffe_weight("pretrained/imagenet_lite_iter_2000000.txt")
        for name, param in model.state_dict().items():
            key = '.'.join(name.split('.')[1:])
            if key not in params_caffe.keys():
                if re.findall("num_batches_tracked",key) == []:
                    print('ERROR miss param: ',name)
                continue
            val = torch.from_numpy(np.reshape(params_caffe[key],param.shape))
            param.copy_(val)
        return
    model = PVANET_LITE(None)
    for k,v in model.state_dict().items():
        print(k,v.shape)
    #model_dict = torch.load("pretrained/imagenet_lite_iter_2000000.pkl")
    #model.load_state_dict(model_dict)
    _load_pva_weight(model)
    model.eval()
    input = torch.from_numpy(np.random.uniform(-128.0,128.0,(4,3,224,224)).astype(np.float32))
    #input = torch.zeros((4,3,224,224),dtype=torch.float)
    outputs = model(input)
    print(f"input: {input.shape}")
    for index,output in enumerate(outputs):
        print(f"output: {index}, {output.shape}")

    #torch.save(model.state_dict(),"imagenet_lite_iter_2000000.pth")
    #model = torch.load("imagenet_lite_iter_2000000.pth")


