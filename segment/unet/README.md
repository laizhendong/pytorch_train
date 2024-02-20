# 参考
[pytorch-unet](https://github.com/milesial/Pytorch-UNet)


# 训练

# 启动训练

        python train.py

## 主要参数

参见 config.py 注释


## 训练集
config.py中的self.datalist指定两个txt文件，分别表示训练集合测试集，txt中每一行表示一个训练样本，
每一行包括两个图像文件，第一个是图片，第二个是标签文件， 如下是两行示例：  

    G:\trainval\images\00013_3_5.bmp G:\trainval\gts\00013_3_5.bmp  
    G:\trainval\images\00040_2_4.bmp G:\trainval\gts\00040_2_4.bmp  

### 标签文件 
标签文件需要是bmp或png格式，尺寸和图片一致，其中每个像素值表示该像素所属的类别(目前最多256类)

# 预测

predict.py负责网络前向，其主要参数参见其注释



# 模型导出  
model2json.py可以导出模型参数，供后续部署