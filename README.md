# DETR for VisDrone Object Detection

This project implements **DETR (Detection Transformer)** for object detection on the **VisDrone2019** dataset. DETR is an end-to-end object detection model that uses Transformers to directly predict object bounding boxes and classes.

![Detection Result](visualization_results/detection_result.png)

*Sample detection result showing cars, people, and other objects with confidence scores*

## 🚁 About VisDrone Dataset

VisDrone2019 is a large-scale benchmark for drone-based computer vision tasks, containing images captured by various drone platforms. The dataset includes 11 object categories commonly found in aerial imagery:

- `ignored-regions` - Areas to be ignored during evaluation
- `pedestrian` - People walking
- `people` - Stationary people  
- `bicycle` - Bicycles
- `car` - Cars
- `van` - Vans
- `truck` - Trucks
- `tricycle` - Tricycles
- `awning-tricycle` - Covered tricycles
- `bus` - Buses
- `motor` - Motorcycles

## 📊 Dataset Statistics

| Split | Images | Annotations | Size |
|-------|--------|-------------|------|
| Train | 6,471 | 390,651 | 64MB |
| Val | 548 | 33,910 | 7.2MB |
| Test | 1,610 | 75,102 | 14MB |

## 🏗️ Project Structure

```
detr_for_VisDrone/
├── VisDrone/                       # VisDrone dataset & annotations
│   ├── VisDrone_COCO/              # COCO format dataset
│   ├── VisDrone2019-DET-*/         # Original dataset splits
│   └── *.json                      # COCO format annotations
├── outputs/                        # Training outputs & checkpoints
├── visualization_results/          # Visualization outputs
├── models/                         # DETR model implementations
├── datasets/                       # Dataset loading & evaluation
├── util/                          # Utility functions
├── main.py                        # Main training script
├── simple_visualize.py            # Simple visualization
├── visualize_results.py           # Advanced visualization
├── test_*.py                      # Testing scripts
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

### Training

Train DETR on VisDrone dataset:

```bash
# Single GPU training
python main.py --coco_path ./VisDrone/VisDrone_COCO --output_dir ./outputs/visdrone_detr_300ep --epochs 300

# Multi-GPU training (8 GPUs)
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --coco_path ./VisDrone/VisDrone_COCO \
    --output_dir ./outputs/visdrone_detr_300ep \
    --epochs 300
```

### Evaluation

Evaluate trained model on validation set:

```bash
python main.py --batch_size 2 --no_aux_loss --eval \
    --resume ./outputs/visdrone_detr_300ep/checkpoint.pth \
    --coco_path ./VisDrone/VisDrone_COCO
```

### Visualization

Generate detection visualizations:

```bash
# Simple visualization
python simple_visualize.py

# Advanced visualization with more options
python visualize_results.py

# Test on test set
python test_on_test_set.py

# Comprehensive testing
python test_all.py
```

## 📈 Training Results

Our DETR model was trained for 100+ epochs on VisDrone with the following performance:

### Training Progress
- **Learning Rate**: 1e-4 (transformer), 1e-5 (backbone)
- **Batch Size**: 2 per GPU
- **Optimizer**: AdamW
- **Training Time**: ~100 epochs completed

### Loss Trends
- **Training Loss**: 39.22 → 18.00 (54% reduction)
- **Validation Loss**: 44.38 → 19.05 (57% reduction)
- **Classification Error**: 52% → 30% (training), 60% → 35% (validation)

### Key Metrics
| Metric | Epoch 0 | Epoch 99 | Improvement |
|--------|---------|----------|-------------|
| Train Loss | 39.22 | 18.00 | -54% |
| Val Loss | 44.38 | 19.05 | -57% |
| Train Class Error | 52.0% | 30.0% | -42% |
| Val Class Error | 60.5% | 35.0% | -42% |

## 🎯 Model Architecture

- **Backbone**: ResNet-50
- **Transformer**: 6 encoder + 6 decoder layers
- **Object Queries**: 100
- **Hidden Dimension**: 256
- **Feed-forward Dimension**: 2048
- **Attention Heads**: 8

## 📋 Usage Examples

### Loading Pretrained Model

```python
import torch
from models import build_model

# Create model
args = create_args()  # Define your arguments
model, criterion, postprocessors = build_model(args)

# Load checkpoint
checkpoint = torch.load('outputs/visdrone_detr_300ep/checkpoint.pth')
model.load_state_dict(checkpoint['model'])
model.eval()
```

### Inference on Custom Image

```python
from PIL import Image
import torchvision.transforms as T

# Load and preprocess image
image = Image.open('your_image.jpg')
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Run inference
with torch.no_grad():
    outputs = model(transform(image).unsqueeze(0))
```

## 🔧 Configuration

Key training parameters can be modified in `main.py`:

```python
# Learning rates
--lr 1e-4                    # Transformer learning rate
--lr_backbone 1e-5           # Backbone learning rate

# Training schedule
--epochs 300                 # Total epochs
--lr_drop 200               # LR drop epoch

# Loss weights
--bbox_loss_coef 5          # Bounding box loss weight
--giou_loss_coef 2          # GIoU loss weight
--eos_coef 0.1              # End-of-sequence loss weight
```

## 📚 Dataset Preparation

The VisDrone dataset has been converted to COCO format for compatibility with DETR:

1. **Original Format**: VisDrone annotation format (txt files)
2. **Converted Format**: COCO JSON format
3. **Directory Structure**: Standard COCO layout (train2017/, val2017/, annotations/)

See `VisDrone/VisDrone_COCO_README.md` for detailed dataset information.

## 🚨 Known Issues & Solutions

### 1. NumPy Compatibility
**Issue**: `np.float` deprecated in newer NumPy versions
**Solution**: Added compatibility code in `datasets/coco_eval.py`

### 2. Memory Usage
**Issue**: High GPU memory usage with large batch sizes
**Solution**: Use smaller batch sizes (batch_size=2) or gradient accumulation

### 3. Training Interruption
**Issue**: Long training times may require interruption
**Solution**: Model automatically saves checkpoints for resuming training

## 🔄 Resuming Training

To continue training from a checkpoint:

```bash
python main.py --resume ./outputs/visdrone_detr_300ep/checkpoint.pth \
    --coco_path ./VisDrone/VisDrone_COCO \
    --output_dir ./outputs/visdrone_detr_300ep \
    --epochs 300
```

## 📄 License

This project is based on the original DETR implementation by Facebook Research. See `LICENSE` file for details.

## 🙏 Acknowledgments

- [DETR: End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr)
- [VisDrone Dataset](http://aiskyeye.com/)
- PyTorch and torchvision teams

## 📞 Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

**Happy Detecting! 🚁✨** 