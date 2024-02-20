# python ./yolov5train/train.py --rect --img 1024 --batch 2 --epochs 80 --data ./configyolo5/xray.yaml --cfg ./configyolo5/yolov5x.yaml --weights ./x-ray-init-weights.pt --name yolov5x_fold0
# python train.py --img 1024 --batch 4 --epochs 40 --data ./data/zhiguan.yaml --cfg ./models/yolov5m.yaml --weights ./weights/yolov5m.pt --name yolov5m
python train.py --img 640 --batch 24 --device 3 --epochs 20 --data ./data/stain.yaml --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt #ssss
