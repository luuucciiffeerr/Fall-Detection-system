# Fall Detection System

A 3D CNN-based fall detection system trained on ~7,000 video clips. Uses a custom **Simple3D CNN** architecture for binary fall/no-fall classification.

**Test Accuracy: 71.03% | F1 Score: 71.09%**

---

## Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 71.03% |
| **F1 Score** | 71.09% |
| **Training Peak** | 91.8% |
| **Model Size** | ~196 MB |

![Confusion Matrix](confusion_matrix.png)
![Training Metrics](training_metrics.png)

---

## How It Works

The system uses a custom **Simple3D CNN** that applies 3D convolutions directly on spatiotemporal video clips:

- Processes **16 frames** per clip at **112x112** resolution
- Two 3D convolutional layers with max pooling for feature extraction
- Fully connected classifier with dropout for regularization
- Binary output: **Fall** or **No Fall**

### Pipeline

```
Video Input -> Frame Extraction (16 frames) -> Resize (112x112) -> Simple3D CNN -> Softmax -> Fall / No Fall
```

---

## Project Structure

```
Fall-Detection-system/
├── train_fall_pipeline.py    # End-to-end training pipeline
├── predict_fall.py           # Inference / prediction script
├── simple3dcnn_fall.pth      # Trained model weights (not in repo)
├── videos_info.csv           # Video metadata catalog
├── train.csv                 # Training split
├── val.csv                   # Validation split
├── test.csv                  # Test split
├── confusion_matrix.png      # Confusion matrix visualization
├── training_metrics.png      # Loss/accuracy training curves
├── requirements.txt          # Python dependencies
└── archive.zip               # Dataset archive (not in repo)
```

---

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- ~10 GB disk space for dataset

### Setup

```bash
# Clone the repository
git clone https://github.com/luuucciiffeerr/Fall-Detection-system.git
cd Fall-Detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
numpy>=1.24.0
```

---

## Training

### Quick Start

1. Place `archive.zip` (your dataset) in the project root
2. Run the pipeline:

```bash
python train_fall_pipeline.py
```

The pipeline automatically:
- Extracts the dataset from `archive.zip`
- Scans videos and builds `videos_info.csv` metadata
- Creates train/val/test splits (60/20/20)
- Trains the Simple3D CNN model
- Saves model weights and evaluation plots

### Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 4 |
| Clip Length | 16 frames |
| Input Resolution | 112 x 112 |
| Max Epochs | 10 |

### Output Files

| File | Description |
|------|-------------|
| `simple3dcnn_fall.pth` | Trained model weights |
| `confusion_matrix.png` | Test set confusion matrix |
| `training_metrics.png` | Loss and accuracy curves |
| `videos_info.csv` | Video metadata catalog |
| `train.csv` / `val.csv` / `test.csv` | Dataset splits |

---

## Inference

### Single Video

```bash
python predict_fall.py path/to/video.mp4
```

### Batch Prediction

```bash
python predict_fall.py --folder path/to/videos/
```

### Custom Threshold

```bash
python predict_fall.py video.mp4 --threshold 0.7
```

### Example Output

```
Using device: cuda
Model loaded: simple3dcnn_fall.pth
Analyzing: test_video.mp4
Prediction: FALL
Fall probability: 0.832
```

### Python API

```python
from predict_fall import FallPredictor

predictor = FallPredictor("simple3dcnn_fall.pth")
prediction, confidence = predictor.predict_single_video("video.mp4")
print(f"{prediction}: {confidence:.2%}")
```

---

## Dataset

| Property | Value |
|----------|-------|
| Total clips | ~6,988 |
| Train set | ~4,193 (60%) |
| Validation set | ~1,398 (20%) |
| Test set | ~1,398 (20%) |
| Classes | 2 (Fall, No_Fall) |

### Data Format

Each video is cataloged in `videos_info.csv`:

```csv
filename,path,num_frames,fps,width,height,duration_sec,label
example_fall.mp4,fall_dataset/Fall/Raw_Video/example_fall.mp4,57,30.0,1920,1080,1.9,1
example_nofall.mp4,fall_dataset/No_Fall/Raw_Video/example_nofall.mp4,91,30.0,1100,1080,3.0,0
```

> Label `1` = Fall, Label `0` = No_Fall

---

## Model Architecture

```
Simple3DCNN(
  features: Sequential(
    Conv3d(3, 32, kernel_size=(3,3,3), padding=1)
    ReLU()
    MaxPool3d((1,2,2))
    Conv3d(32, 64, kernel_size=(3,3,3), padding=1)
    ReLU()
    MaxPool3d((2,2,2))
  )
  classifier: Sequential(
    Flatten()
    Linear(64 * 8 * 28 * 28, 128)
    ReLU()
    Dropout(0.5)
    Linear(128, 2)
  )
)
```

---

## Tech Stack

- **Deep Learning**: PyTorch
- **Video Processing**: OpenCV
- **Data Management**: pandas, NumPy
- **Evaluation**: scikit-learn
- **Visualization**: matplotlib, seaborn

---

## Authors

- **Morteza Mohasebati** — [GitHub](https://github.com/Mortezamohasebati)
- **Ali Abroudoust** — [GitHub](https://github.com/luuucciiffeerr)

---

## License

This project is licensed under the MIT License.
