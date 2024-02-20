from yacs.config import CfgNode as CN

_C = CN()

##############################################
#Model
_C.MODEL = CN()
_C.MODEL.BACKBONE_NAME = ""
_C.MODEL.WEIGHTS = ""
_C.MODEL.USE_IMAGENET_BACKBONE = True

############################################
#Dataset
_C.DATA = CN()
_C.DATA.SRC_DIR = ""
_C.DATA.TRAIN_LIST = ""
_C.DATA.TEST_LIST = ""
_C.DATA.NAME = ""
_C.DATA.CLASSES_BALANCED = False
_C.DATA.MEAN = (0,0,0)
_C.DATA.STD = (1,1,1)
#Augment
_C.DATA.AUGMENT = CN()
_C.DATA.AUGMENT.SHEAR = (0,0)
_C.DATA.AUGMENT.BLUR = -1
_C.DATA.AUGMENT.GAMMA = 0.0
_C.DATA.AUGMENT.ROTATION = -1.0
_C.DATA.AUGMENT.HFLIP = False
_C.DATA.AUGMENT.VFLIP = False
_C.DATA.AUGMENT.COLOR = False
_C.DATA.AUGMENT.SHUFFLE_COLOR = False
_C.DATA.AUGMENT.RESIZE = (0,0) #W,H
_C.DATA.AUGMENT.CROP = (0,0) #W,H
_C.DATA.AUGMENT.RandomBrightness = 0.0
_C.DATA.AUGMENT.RandomContrast = [1.0, 1.0]
######################################
#SOLVER
_C.SOLVER = CN()
_C.SOLVER.LR_BASE = 0.001
_C.SOLVER.LR_POLICY = "cosine"
_C.SOLVER.GRADIENT_MAX = -1
_C.SOLVER.EPOCHS = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.OUTPUT_FOLDER = ""
_C.SOLVER.DEVICE = 0
_C.SOLVER.STEP = 0

######################################
#ONNX
_C.ONNX = CN()
_C.ONNX.INPUT_SHAPE = [1, 3, 224, 224]
_C.ONNX.NUM_CLASSES = -1
_C.ONNX.CLASSES = []
_C.ONNX.PTH_NAME = 'torch.pth'
_C.ONNX.ONNX_NAME = 'onnx.onnx'
_C.ONNX.INPUT_NAME = ['data']
_C.ONNX.OUTPUT_NAME = ['prob']
_C.ONNX.DYNAMIC_SHAPE = True

def get_default_config():
    return _C.clone()


