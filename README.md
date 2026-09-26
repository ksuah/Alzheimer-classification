# Alzheimer's Stage Classification

Classification of Alzheimer's disease stages from MRI images with Grad-CAM visualization.

## Overview
This project implements a full pipeline for classifying Alzheimer's disease stages from MRI scans:
data loading, training with transfer learning, evaluation, inference, and model interpretability via Grad-CAM.

## Stack
- Python, PyTorch, torchvision
- ResNet18 (transfer learning)
- Grad-CAM for model interpretability
- OpenCV, NumPy, PIL

## Structure
- `src/dataset.py` — custom Dataset and DataLoader
- `src/model.py` — ResNet18-based classifier
- `src/train.py` — training loop, validation, checkpointing
- `src/interface.py` — inference on a folder of images
- `src/gradcam.py` — Grad-CAM visualization of model attention
- `split_dataset.py` — train/val/test split

## Classes
- Non_Demented
- Very_Mild_Demented
- Mild_Demented
- Moderate_Demented

## Results
- Best model selected by validation accuracy
- Grad-CAM highlights regions the model relies on for prediction

## Usage
1. Install dependencies: `pip install torch torchvision pillow opencv-python numpy`
2. Prepare data and split it with `split_dataset.py`
3. Train: `python src/train.py`
4. Inference: `python src/interface.py`
5. Grad-CAM: `python src/gradcam.py`

## Notes
- Dataset and trained weights are not included in the repository.
- The model is based on a pretrained ResNet18 fine-tuned for 4-class classification.