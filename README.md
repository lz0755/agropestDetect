# AgroPest Detection using YOLO

## Description

This project implements pest detection in agricultural images using YOLO (You Only Look Once) object detection models. The notebook `yolodepest.ipynb` demonstrates training, prediction, and evaluation of various YOLO models (YOLOv8s, YOLOv11s, YOLOv12s) on a custom dataset for identifying pests such as ants.

## Requirements

- Python 3.8 or higher
- ultralytics library (install via `pip install ultralytics`)
- PyTorch (automatically installed with ultralytics)
- CUDA-compatible GPU (optional, for faster training)

## Dataset

The dataset is organized in the `data/` folder with the following structure:
- `data.yaml`: Dataset configuration file specifying paths to train, validation, and test sets, and class names.
- `train/`: Training images and labels
- `valid/`: Validation images and labels
- `test/`: Test images and labels

The dataset contains images of agricultural pests, primarily ants, with bounding box annotations in YOLO format.

## Installation

1. Clone or download this repository.
2. Install dependencies:
   ```
   pip install ultralytics
   ```
3. Ensure you have the YOLO model weights (yolo12s.pt, yolov8s.pt, yolo11s.pt, yolo11n.pt) in the root directory.

## Usage

### Training

Open `yolodepest.ipynb` in Jupyter Notebook or JupyterLab. Run the training cell to train a YOLO model:

```python
from ultralytics import YOLO

model = YOLO("yolo12s.pt")

model.train(
    data="data/data.yaml",
    epochs=100,
    patience=25,
    imgsz=640,
    batch=8,
    lr0=0.005,
    device=0,
    project="training-iteration",
    name="yolo12s",
    save=True,
    save_period=5,
    # Data augmentation parameters...
)
```

This will train the model and save weights in `training-iteration/yolo12s/weights/`.

### Prediction

Use the trained model for inference on test images:

```python
model = YOLO("training-iteration/yolo12s/weights/best.pt")
model.predict(
    source="data/test/images",
    conf=0.45,
    batch=16,
    save=True
)
```

Predictions will be saved in `runs/detect/predict/`.

### Evaluation

Evaluate the trained model:

```python
metrics = model.val()
print(metrics)
```

This computes metrics like mAP, precision, recall, etc.

## Models

The project includes training and evaluation of multiple YOLO variants:
- YOLOv8s: Baseline model
- YOLOv11s: Improved version
- YOLOv12s: Latest version

Trained weights are stored in `training-iteration/` subfolders.

## Results

Training results, predictions, and validation outputs are saved in the `runs/` folder:
- `runs/detect/`: Prediction results
- `runs/val/`: Validation results

Check `results.csv` in each training iteration for detailed metrics.

## Contributing

Feel free to contribute by improving the models, adding more pest classes, or optimizing training parameters.

## License

This project is open-source. Please check individual model licenses for YOLO weights.
