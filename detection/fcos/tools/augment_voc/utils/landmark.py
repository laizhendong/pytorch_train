import cv2,yaml,re

def load(path):
    with open(path,'r') as f:
        content = '\r\n'.join(f.readlines()[1:])
        content = re.sub(':',' : ',content)
        landmark = yaml.load(content,Loader=yaml.FullLoader)
    return landmark['tube_landmarks'],landmark['silk_landmarks']