#########Train with multiple(4) GPUs. (DDP Mode)
#python -m torch.distributed.launch --nproc_per_node 4 train.py --device 0,1,2,3
#########Train with specified GPUs. (for example with GPU=3)
#python train.py --device 3
#########Train the orignal dataset demo
python train.py --data 'data/yolov5obb_bansi.yaml' --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt --epochs 150 --batch-size 6 --img 1024 --multi-scale --device 3 --name yolov5s_bansi_multi_scale
#########Train the splited dataset demo
#python train.py --data 'data/yolov5obb_demo_split.yaml' --epochs 10 --batch-size 2 --img 1024 --device 1
#########Train the splited dataset DOTA
#python train.py --data 'data/dotav15_poly.yaml' --weights ./weights/yolov5m.pt --epochs 150 --batch-size 4 --img 1024 --multi-scale --device 3 --name yolov5m