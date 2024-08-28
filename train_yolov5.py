import torch

# Path to the YOLOv5 repository
YOLOV5_REPO = 'https://github.com/ultralytics/yolov5'

# Clone YOLOv5 repository if not already cloned
!git clone {YOLOV5_REPO} yolov5
%cd yolov5

# Install requirements
!pip install -r requirements.txt

# Define training parameters
DATA_CFG = 'data/weapon_detection.yaml'  # Path to dataset YAML file
EPOCHS = 50  # Number of training epochs

# Run training
!python train.py --img-size 640 --batch-size 16 --epochs {EPOCHS} --data {DATA_CFG} --weights yolov5s.pt --device 0
