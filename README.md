# Threat Identification System

Real-time AI-powered security surveillance system that detects weapons (knives, guns, rifles) and identifies known individuals using computer vision, with automated alerting via email and audio.

## Features

- **Weapon Detection** — YOLOv5-based object detection trained on custom weapon dataset (knife, gun, rifle categories)
- **Face Recognition** — DeepFace with FaceNet embeddings for real-time identity verification against a known persons database
- **Automated Alerts** — Sound alarm + email notification when threats are detected
- **GUI Interface** — PyQt5 desktop application with live camera feed and detection overlays

## Architecture

```
Camera Feed ──> YOLOv5 (weapon detection) ──> Alert System (email + sound)
     |
     └──> DeepFace (face recognition) ──> Identity Match
```

## Tech Stack

`Python` `PyTorch` `YOLOv5` `DeepFace` `OpenCV` `PyQt5`

## Files

| File | Description |
|------|-------------|
| `index.py` | Main application — GUI, camera feed, detection loop, alerting |
| `train_yolov5.py` | YOLOv5 training script for custom weapon detection |

## Usage

```bash
pip install torch torchvision opencv-python deepface PyQt5 playsound
python index.py
```

## Training

The weapon detector is fine-tuned from YOLOv5s pretrained weights on a custom dataset:

```bash
python train_yolov5.py  # 50 epochs, 640px, batch 16
```

## License

MIT
