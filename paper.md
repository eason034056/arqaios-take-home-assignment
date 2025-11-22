
# 📄 **全文文字提取（原文；共 19 頁）**

來源：*Human_identifiaction.pdf*

---

## **PAGE 1**

**CHAPTER 6
HUMAN IDENTIFICATION USING MMWAVE RADAR**

This chapter introduces an innovative approach using deep learning for human iden­ti­fication utilizing mmWave radar technology. Unlike conventional vision methods, the approach in this work ensures privacy and accuracy in various indoor settings. Lever­ag­ing partial PointNet, MLP, CNN, and Bi-LSTM network components, this chapter proposes a unique neural network structure named MMIDNet, designed to process point cloud data from mmWave radar directly. The system achieves an impressive identification accuracy of 92.4% for 12 individuals. The research encompasses initial evaluation, data collection, system design, and system evaluation, highlighting the potential of mmWave radar combined with deep learning for secure and efficient human identification in IoT applications. The content of this chapter has been partially published in the MECO conference 2024 [53].

The sections are organized as follows: Section 6.1 provides a brief overview of this chapter. Section 6.2 evaluates the human identification task using the FAUST dataset, demonstrating the feasibility of using deep learning techniques to identify human targets by using point clouds. Following this, Section 6.3 details the preparation and pre-processing of the point cloud dataset collected using mmWave radar. Section 6.4 presents the specifics of the network structure and explains its contribution to the human identification problem. Section 6.5 provides a comprehensive evaluation of the system, covering overall performance as well as the effects of different network configurations, observation durations, and numbers of people. It also includes a comparison with other researchers’ works. Finally, Section 6.6 provides a conclusion for this chapter.

---

## **PAGE 2**

### **6.1 Overview**

In the evolving landscape of sensing technologies, the precise identification of human targets within indoor environments stands as a critical endeavour with profound implications for various applications. Whether in the realms of in-house security, healthcare, or smart building management, the accurate and unobtrusive identification of individuals within indoor spaces is essential. Traditional sensing methods, including optical cameras and infrared sensors, have been foundational in this pursuit. However, these methods frequently encounter challenges like limited visibility, difficulties in adapting to varying lighting conditions, and privacy concerns.

To overcome these challenges, the emergence of mmWave radar technology offers an innovative solution for indoor human target identification. Many researchers explored applications of human identification using point clouds obtained by mmWave radar.

In this study, three FMCW radar IWR1843 development boards from TI are utilized to generate point clouds within indoor environments for training, validation, and test datasets. This chapter explores the potential of harnessing mmWave radar technology for human identification and proposes a deep neural network tailored for datasets in point cloud format to enhance accuracy.

Primary contributions:

• Feasibility evaluation using FAUST dataset
• Proposal of MMIDNet neural network
• Full system evaluation showing 92.4% accuracy
• Public GitHub repository

### **6.2 FAUST Evaluation**

Human identification typically requires abundant information like facial features; however, mmWave radar point clouds are sparse. Therefore, FAUST is used first to validate feasibility.

The FAUST dataset consists of 10 subjects × 30 static postures = 100 mesh models generated from multi-camera high-resolution systems.

---

## **PAGE 3**

### **Point Cloud Generation**

mmWave radar outputs (x,y,z) points; thus, FAUST meshes must be sampled accordingly. Because radar frames have <200 points, FAUST meshes are uniformly sampled to 100–200 points per sample, 100 samples per mesh → total 10,000 samples.

Neural networks need consistent point counts → zero-pad everything to 200 points → tensor shape:

```
X ∈ R^(N × P × C)
N = 10000, P = 200, C = 3
```

### **Data Augmentation**

To simulate realistic indoor placement:

• Random translation (0–3m) along x,y
• Random rotation (0–360°) along z
• Rigid transform to preserve body shape
• Stratified split: 70% train / 10% val / 20% test

### **Network Design**

Two architectures evaluated:
(1) MLP
(2) CNN + MLP

Implemented in Keras 2.6.0, Python 3.8.6.

---

## **PAGE 4**

### **MLP**

Two configurations:
• 5-layer MLP → 3.6M params
• 3-layer MLP → 1.1M params

Input = 200×3 → flatten → 600 → dense layers → softmax(10)

### **CNN + MLP**

Three configurations:
• 5 convolution layers (2.27M params)
• 3 conv layers (824k params)
• 1 conv layer (216k params)

Conv1D with specified kernel count & size → MaxPooling → Flatten → Dense head.

---

## **PAGE 5**

### **Results and Comparison**

Training:

• Loss: cross entropy
• Optimizer: Adam
• Epochs: 300
• LR = 0.0002
• Batch = 128

MLP:
• Accuracy plateaus at ~40%
• Overfits after epoch 160

CNN + MLP:
• 1 or 3 conv layers → ~70% accuracy
• 5 conv layers → ~80% accuracy (best)

Conclusion: CNN+MLP > MLP; shows that sparse point clouds still contain identity information.

---

## **PAGE 6–7**

### **6.3 Data Preparation**

#### **Data Collection**

12 individuals
10 rounds × 1–2 min walking
~20 min per participant
~288,000 frames total
Ethics-approved
Video used only for labeling

#### **Data Preprocessing**

Point format:

```
(x, y, z, v, snr)
```

Steps:

1. Convert all radar local coordinates into global frame
2. Filter by position & SNR
3. DBSCAN clustering → extract human point cluster
4. Zero-pad each frame to 200×5
5. Sliding window:
   • 30 frames
   • step = 5
   • overlap = 83%
   → sample = 30×200×5

Dataset shape:

```
X ∈ R^(N × T × P × C)
T=30, P=200, C=5
```

Everything stored in float16.

---

## **PAGE 8–12**

## **6.4 System Design**

MMIDNet = 5 blocks:

1. Transformation Block (T-Net for (x,y,z))
2. Residual CNN Block
3. Global Max Pooling (permutation invariance)
4. Bi-LSTM Block (gait & temporal information)
5. Dense Block (classification)

Key ideas:

* T-Net predicts 3×3 affine matrix
* Residual CNN avoids degradation
* MaxPooling makes permutation-invariant
* LSTM models motion sequence
* Dense layers include dropout to reduce overfitting

---

## **PAGE 13**

### **Network Training**

Environment:

* Python 3.8.6
* TensorFlow 2.6.5
* RTX 2080 (local)
* A100 40GB on Colab (final training)

Hyperparameters:

* Batch size 128
* LR 0.0002
* Epochs 200

### **Sliding Window Prediction**

Use a 12-sample window (3 seconds) → majority vote to correct mislabels → greatly improves final accuracy.

---

## **PAGE 14–15**

## **6.5 System Evaluation**

### **6.5.1 Identification Performance**

MMIDNet test results:

* Accuracy = **92.4%**
* F1 = **92.0%**

Confusion matrix shows high per-class accuracy, with some confusion between similar body shapes (e.g., subject 11 vs 5).

### **6.5.2 Network Structure Evaluation**

Component contribution:

* CNN+Dense only → 61.1%
* +T-Net & Global Max Pooling → 70.5%
* +Bi-LSTM → 73.0%
* All combined → 86.3%
* +Sliding window → **92.4%**

---

## **PAGE 16–17**

### **6.5.3 Observation Duration**

Longer observation → higher accuracy:

* 3 seconds → 92.4%
* 7 seconds → >98%

### **6.5.4 Number of People**

Accuracy by group size:

* 4 people → 98%
* 6 people → ~97%
* 12 people → 92–95%

Significantly stronger than WiFi CSI methods.

---

## **PAGE 18–19**

### **6.5.5 System Comparison**

Compared to:

| Technology             | Accuracy  | Notes                                |
| ---------------------- | --------- | ------------------------------------ |
| Camera                 | >98%      | Severe privacy issues                |
| RGB-D                  | 98.4%     | Still privacy issues                 |
| Floor sensor           | 80%       | Hard to deploy                       |
| WiFi CSI               | 80%       | Requires motion between TX/RX        |
| Voxelized mmWave (mID) | 89%       | Heavy processing                     |
| **MMIDNet**            | **92.4%** | Sparse point cloud; privacy-friendly |

### **6.6 Conclusion**

MMIDNet:

* Processes sparse mmWave point clouds
* Achieves 92.4% accuracy (3s)
* Achieves >98% accuracy (7s)
* Supports multi-person environments
* Efficient & privacy-preserving

Future work:

* Larger datasets
* Multi-person classification
* More complex environments

