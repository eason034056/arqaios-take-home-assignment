# mmWave Radar Human Identification - POC Implementation

A PyTorch implementation of point cloud-based human identification models, inspired by the MMIDNet research paper. This project demonstrates feasibility of using sparse 3D point clouds for person identification, serving as a proof-of-concept for privacy-preserving indoor human sensing.

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## 🎯 Problem Statement

Traditional human identification systems rely on cameras, raising significant privacy concerns. This project explores using **mmWave radar point clouds** for human identification—a privacy-preserving alternative that:

- Does not capture facial features or detailed imagery
- Works in low-light and through occlusions
- Generates sparse 3D point clouds (typically <200 points per frame)
- Identifies individuals based on body shape and gait patterns

**Task**: Given a sparse 3D point cloud representing a person, classify which individual (out of 10 subjects) it represents.

---

## 📊 Dataset

### FAUST Dataset

We use the **FAUST (Faces of Articulated Super-humans Through Training)** dataset for initial feasibility evaluation:

- **Source**: MPI FAUST dataset (100 watertight meshes)
- **Subjects**: 10 individuals
- **Poses**: 10 different poses per subject
- **Samples**: 100 meshes × 100 samples per mesh = 10,000 point cloud samples
- **Format**: Each sample has 200 points × 3 coordinates (x, y, z)

### Data Processing Pipeline

```
Mesh (.ply/.obj) → Uniform Sampling → 200 Points → Augmentation → Model Input
```

**Preprocessing steps**:
1. **Sampling**: Convert mesh to 200-point cloud using Farthest Point Sampling (FPS)
2. **Normalization**: Scale to unit sphere for consistent scale
3. **Augmentation** (training only):
   - Random rotation (0-360° around z-axis)
   - Random translation (±3m in x-y plane)
   - Rigid transformations to preserve body shape

**Data Split**:
- Train: 70% (7,000 samples)
- Validation: 10% (1,000 samples)
- Test: 20% (2,000 samples)
- Stratified by subject to maintain class balance

---

## 🧠 Approach

This POC implements and compares **three architectures** of increasing sophistication:

### 1. MLP Baseline
- Simple fully-connected network
- Flattens point cloud to 1D vector (600 features)
- **Purpose**: Establish baseline; verify data contains identity information
- **Limitation**: Order-dependent, no geometric understanding

### 2. 1D-CNN Model
- Applies 1D convolutions across sorted points
- Captures local spatial patterns
- Uses global max pooling for partial permutation invariance
- **Purpose**: Show that local features improve performance
- **Limitation**: Relies on sorting, which is somewhat arbitrary

### 3. Tiny PointNet (Best)
- Inspired by PointNet architecture
- **Key innovations**:
  - T-Net: Learns spatial transformation for alignment
  - Shared MLP: Processes each point independently
  - Global Max Pooling: Achieves true permutation invariance
- **Purpose**: Demonstrate state-of-the-art point cloud processing
- **Advantages**: Order-invariant, learns geometric features

---

## 📁 Project Structure

```
arqaios-take-home-assignment/
│
├── README.md                   # This file
├── config.yaml                 # Configuration (hyperparameters, paths)
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── raw/                    # FAUST mesh files (.ply/.obj)
│   └── processed/              # Preprocessed point clouds (.npz)
│
├── src/
│   ├── preprocessing.py        # Data augmentation & normalization
│   ├── dataset.py              # PyTorch Dataset & DataLoader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py              # MLP Baseline
│   │   ├── cnn1d.py            # 1D-CNN Model
│   │   └── pointnet_tiny.py    # Tiny PointNet
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation & visualization
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
│
└── results/
    ├── checkpoints/            # Saved model weights
    ├── tensorboard/            # TensorBoard logs
    ├── confusion_matrix.png    # Confusion matrices
    └── model_comparison.csv    # Performance comparison table
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended, but CPU works)
- 8GB+ RAM

### Setup

```bash
# Clone repository
cd arqaios-take-home-assignment

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download FAUST dataset
# Place mesh files (.ply or .obj) in data/raw/
# Download from: http://faust.is.tue.mpg.de/
```

---

## 🚀 Usage

### 1. Data Preprocessing

The first training run will automatically process the raw FAUST meshes:

```bash
# Data will be processed and saved to data/processed/faust_pc.npz
# This happens automatically during first training run
```

### 2. Training Models

Train each model architecture:

```bash
# Train MLP Baseline
python src/train.py --config config.yaml --model mlp

# Train 1D-CNN Model
python src/train.py --config config.yaml --model cnn1d

# Train Tiny PointNet
python src/train.py --config config.yaml --model pointnet
```

**Training options**:
- `--config`: Path to config file (default: `config.yaml`)
- `--model`: Model type (`mlp`, `cnn1d`, or `pointnet`)
- `--resume`: Path to checkpoint for resuming training

**What happens during training**:
1. Loads/processes FAUST dataset
2. Creates train/val/test splits
3. Initializes model and optimizer
4. Trains for specified epochs with early stopping
5. Logs metrics to TensorBoard
6. Saves best model checkpoint

**Monitor training**:
```bash
# Launch TensorBoard
tensorboard --logdir results/tensorboard

# View at http://localhost:6006
```

### 3. Evaluation

Evaluate a single model:

```bash
python src/evaluate.py \
    --model pointnet \
    --checkpoint results/checkpoints/pointnet/model_best.pth
```

Compare all three models:

```bash
python src/evaluate.py --compare --models mlp cnn1d pointnet
```

**Evaluation outputs**:
- Test accuracy and F1-score
- Confusion matrix (counts and percentages)
- Per-class precision/recall/F1
- Model comparison table (CSV)

### 4. Exploratory Data Analysis

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/eda.ipynb
```

The EDA notebook visualizes:
- 3D point clouds from different subjects
- Augmentation effects
- Class distribution
- Dataset statistics

---

## 🏗️ Model Architectures

### MLP Baseline

```
Input (B, 200, 3) → Flatten (B, 600)
                  → Dense(256) → BN → ReLU → Dropout(0.3)
                  → Dense(128) → BN → ReLU → Dropout(0.3)
                  → Dense(10)
```

**Parameters**: ~200K  
**Key Issue**: Order-dependent; treats 3D coordinates as arbitrary features

---

### 1D-CNN Model

```
Input (B, 200, 3) → Sort by z-coord → Transpose (B, 3, 200)
                  → Conv1D(64) → BN → ReLU
                  → Conv1D(128) → BN → ReLU
                  → Conv1D(256) → BN → ReLU
                  → GlobalMaxPool → (B, 256)
                  → Dense(128) → Dropout → Dense(10)
```

**Parameters**: ~300K  
**Improvement**: Captures local spatial patterns; max pooling provides some invariance

---

### Tiny PointNet

```
Input (B, 200, 3) → Transpose (B, 3, 200)
                  → T-Net (predicts 3×3 transformation)
                  → Apply Transformation
                  → Conv1D(64) → BN → ReLU
                  → Conv1D(128) → BN → ReLU
                  → Conv1D(1024) → BN → ReLU
                  → GlobalMaxPool → (B, 1024)
                  → Dense(512) → BN → ReLU → Dropout(0.3)
                  → Dense(256) → BN → ReLU → Dropout(0.3)
                  → Dense(10)
```

**Parameters**: ~1.5M  
**Key Features**:
- **T-Net**: Learns canonical alignment (rotation/translation invariance)
- **Shared MLP**: Processes points independently (permutation invariance)
- **Global Max Pooling**: Aggregates features symmetrically

---

## 📈 Results

### Expected Performance (based on paper and roadmap)

| Model         | Val Acc | Test Acc | Parameters | Key Characteristics              |
|---------------|---------|----------|------------|----------------------------------|
| MLP Baseline  | ~35-40% | ~30-40%  | ~200K      | Order-dependent; weak baseline   |
| 1D-CNN        | ~65-70% | ~60-70%  | ~300K      | Captures local patterns          |
| Tiny PointNet | ~80-85% | ~75-85%  | ~1.5M      | Best; permutation-invariant      |

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Macro F1-Score**: Average F1 across all classes (treats classes equally)
- **Confusion Matrix**: Shows which subjects are confused with each other
- **Per-Class Metrics**: Precision, recall, F1 for each subject

### Training Curves

Training typically shows:
- MLP: Quick plateau around 40% (overfits easily)
- 1D-CNN: Steady improvement to 65-70%
- PointNet: Best performance, 75-85% (may require more epochs)

---

## 🔧 Configuration

Key hyperparameters in `config.yaml`:

```yaml
# Data
data:
  num_points: 200           # Points per sample
  
# Training
training:
  batch_size: 64
  num_epochs: 120
  learning_rate: 0.0002
  weight_decay: 0.0001
  early_stopping_patience: 20

# Model
model:
  dropout: 0.3              # Regularization

# Augmentation
augmentation:
  rotation_range: 360       # Degrees
  translation_range: 3.0    # Meters
```

---

## ⚠️ Limitations & Future Work

### Current Limitations

1. **Static Pose Only**: Uses single-frame point clouds; doesn't leverage temporal information
2. **Small Dataset**: 10 subjects may not generalize to real-world scenarios
3. **Synthetic Data**: FAUST meshes don't have mmWave noise characteristics
4. **No Multi-Person**: Assumes one person per sample
5. **No DBSCAN Clustering**: Real mmWave data requires point cloud segmentation

### Future Improvements

**Near-term** (to match MMIDNet):
- [ ] Add temporal modeling (Bi-LSTM for 30-frame sequences)
- [ ] Implement multi-radar fusion
- [ ] Add DBSCAN clustering for person segmentation
- [ ] Include velocity (Doppler) information
- [ ] Test on real mmWave radar data

**Long-term**:
- [ ] Scale to 50+ subjects
- [ ] Multi-person simultaneous identification
- [ ] Online learning for new subjects
- [ ] Deployment on edge devices (model compression)
- [ ] Adversarial robustness testing

---

## 📚 References

### Papers

1. **Original Paper**: "Human Identification Using mmWave Radar" (MECO 2024)
   - Proposes MMIDNet architecture
   - Achieves 92.4% accuracy on 12 subjects
   - Uses T-Net, Residual CNN, and Bi-LSTM

2. **PointNet**: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" (CVPR 2017)
   - Introduces permutation-invariant architecture
   - T-Net for transformation alignment
   - https://arxiv.org/abs/1612.00593

3. **FAUST Dataset**: "FAUST: Dataset and Evaluation for 3D Mesh Registration" (CVPR 2014)
   - 100 human body meshes
   - 10 subjects × 10 poses
   - http://faust.is.tue.mpg.de/

### Key Concepts

- **Permutation Invariance**: f({p1, p2, p3}) = f({p2, p3, p1})
- **Farthest Point Sampling (FPS)**: Maintains shape better than random sampling
- **T-Net**: Learns optimal spatial transformation for canonical alignment
- **mmWave Radar**: 77-81 GHz frequency; generates sparse 3D point clouds

---

## 🤝 Acknowledgments

- **Arqaios**: For the take-home assignment opportunity
- **TI IWR1843**: mmWave radar hardware used in original research
- **MPI FAUST**: For providing the human mesh dataset

---

## 📝 Code Explanation Philosophy

All code in this repository follows these principles:

1. **Extensive Comments**: Every function has detailed docstrings explaining:
   - What the function does (purpose)
   - How it works (algorithm)
   - Parameters and return values
   - Usage examples

2. **Named Clearly**: Function and variable names describe their purpose:
   - `normalize_to_unit_sphere()`: clearly states it scales to [-1, 1]
   - `stratified_split()`: indicates class-balanced splitting
   - `TNet`: T-Net from PointNet paper (transformation network)

3. **Step-by-Step**: Complex operations are broken into numbered steps with comments

4. **Why, Not Just What**: Comments explain design decisions:
   - Why use FPS over random sampling? (Better shape preservation)
   - Why Global Max Pooling? (Permutation invariance)
   - Why sort before 1D-CNN? (Establish consistent ordering)

---

## 📧 Contact

For questions about this implementation:
- Review the code comments (most questions answered inline)
- Check the EDA notebook for data insights
- Refer to the original paper for theoretical background

---

## 📄 License

This project is for educational and research purposes. Please cite the original MMIDNet paper if using this work academically.

---

**Built with** ❤️ **using PyTorch**
