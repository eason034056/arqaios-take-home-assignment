

# ✅ **Arqaios AIML Engineer Round 2 — Full Project Roadmap**

### 規模

* **總時程：4–8 小時（官方建議）**
* **產出：Technical Summary + POC Code + 可解釋的模型選擇與結果**

---

# 0. 專案總覽（你要在面試講得出來的）

這是一個 **Research → Engineering → Clean POC** 的作業。
你的目標不是 accuracy，而是：

* 你能不能讀懂 mmWave + point cloud 的困難點
* 你如何設計一個 **資料處理 → 模型 → 訓練 → 評估** 的 pipeline
* 你能不能 **清楚、合理地 justify** 你的方法
* 你的 code 是否乾淨、模組化
* 你的報告是否能站在研究者 / Engineer 的角度說明 trade-offs

---

# 1. **第 0 步：建立 Repo 結構（0.5 小時）**

推薦使用以下目錄（最專業、最乾淨的 POC 標準）：

```
root/
│── README.md
│── technical_summary.pdf
│── requirements.txt
│── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── src/
│   ├── dataset.py        # load FAUST → point cloud → sampling → padding → augmentation
│   ├── preprocessing.py  # rotation / translation / normalize
│   ├── models/
│   │     ├── mlp.py
│   │     ├── cnn1d.py
│   │     ├── pointnet_tiny.py
│   │
│   ├── train.py          # loop, logs
│   ├── evaluate.py       # accuracy, confusion matrix
│
├── notebooks/
│   ├── eda.ipynb         # visualize point cloud, augmentation check
│
└── results/
    ├── curves.png
    ├── confusion_matrix.png
    ├── comparison_table.csv
```

---

# 2. **閱讀論文與整理 Technical Summary（1–1.5 小時）**

Technical Summary（1–2 pages）依照官方要求，需包含：

### (1) 問題定義

* mmWave 雷達 + Point cloud
* 隱私保護
* 室內 human identification

### (2) mmWave 為何適合

* 不依賴光線
* 不拍攝影像（privacy-friendly）
* 低 input size（比 voxel 少很多）
* 能捕捉 **形狀 + gait**

### (3) Dataset（FAUST + mmWave pipeline）

* FAUST：10 人、mesh → sampled 100–200 點
* mmWave：多雷達 → 對齊 → filter → DBSCAN 分 cluster
* 之後做：

  * 時間窗切片
  * Zero-padding
  * Normalize

### (4) MMIDNet 的設計重點

* T-Net（姿態/旋轉不變性）
* Residual CNN（形狀特徵）
* Global Max Pooling（permutation invariant）
* Bi-LSTM（時間序 gait）

你在 POC **不用重現 MMIDNet**，但要能說清楚哪些精神你有保留。

### (5) Strength / Limitation

* Sparse → 小模型即可
* 多雷達強化 robustness
* mmWave noisy → preprocessing 很重要
* 小 dataset → 容易 overfitting

完成後輸出 **technical_summary.pdf**。

---

# 3. **Dataset 建立（1 小時）**

你會選 FAUST（官方允許，且論文 6.2 已示範）。

### 3.1 下載 FAUST

* 從 MPI FAUST 官方（已提供）
* 100 watertight meshes
* 每人 10 姿勢 → 10 類別分類問題

### 3.2 轉 mesh → point cloud

程式流程：

```
mesh → uniformly sample 150–200 surface points
```

所有 sample shape：

```
P × C = (200 × 3)
```

### 3.3 Fixed-length sampling

* 若 >200：farthest point sampling or random sampling
* 若 <200：zero-padding

### 3.4 Data augmentation（至少做 2 項）

符合論文：

* random rotation (0–360° around z)
* random translation (x,y shift)
* normalization to unit sphere

### 3.5 Split

```
Train 70%
Val   10%
Test  20%
```

同 identity 分開（subject-wise split）。

---

# 4. **Modeling（1 小時）**

POC 最少 1 種，你推薦做 3 種（會讓你特別強）：

---

## ✔ 模型 A：MLP baseline（很快建立 sanity check）

結構：

```
Flatten → Dense(256) → BN → ReLU → Dropout  
        → Dense(128) → BN → ReLU → Dropout  
        → Dense(num_classes, Softmax)
```

意義：

* 檢查資料本身是否帶 identity cue
* 準備被 CNN / PointNet 超越（需要寫在報告 justify）

---

## ✔ 模型 B：1D-CNN（從 MCIDNet 殘差 CNN 簡化）

做法：先把點依某個維度排序，例如 z 軸，然後：

```
Conv1D(64,1) → BN → ReLU  
Conv1D(128,1) → BN → ReLU  
GlobalMaxPooling  
Dense head
```

優點：

* 捕捉鄰近關係
* 訓練比 MLP 好很多

---

## ✔ 模型 C：Tiny PointNet-like（最強、最符合論文）

```
Conv1D(64,1) → BN → ReLU  
Conv1D(128,1) → BN → ReLU  
GlobalMaxPooling
Dense(128) → Dropout  
Dense(num_classes)
```

如果要更強，可以加「簡化版 T-Net」：

* 輸入 → Conv1D(32,1) → FC → 3×3 transformation matrix → apply to xyz

但非必要。

---

# 5. **Training（0.5–1 小時）**

設定通用參數：

```
Loss: CrossEntropy
Optimizer: Adam(lr=2e-4)
Batch size: 64
Epochs: 80–120
EarlyStopping
```

輸出：

* train_acc / val_acc curve
* train_loss / val_loss curve
* 混淆矩陣

存放到：

```
results/curves.png
results/confusion_matrix.png
```

---

# 6. **Evaluation（0.5 小時）**

你應該要比較：

| Model         | Val Acc | Test Acc | Params | Key notes             |
| ------------- | ------- | -------- | ------ | --------------------- |
| MLP           | ~30–40% | 低        | 高      | 順序敏感、忽略 geometry      |
| 1D-CNN        | ~60–70% | 中        | 中      | 有局部特徵但仍排依賴            |
| Tiny PointNet | ~75–85% | 最高       | 低      | permutation-invariant |

這張圖表在你面試時會超強。

---

# 7. **寫 POC 內容（0.5 小時）**

寫 README？只要包含：

### (1) Problem

### (2) Dataset

### (3) Preprocessing

### (4) Model Choice + Justification

### (5) Results

### (6) Limitations & Next steps

Next steps 建議寫：

* 加入時序資訊（T=30 frames）→ 變成 gait-based
* 多雷達 alignment
* DBSCAN cluster 分人
* 加 T-Net 做 transformation invariance
* 跟 MMIDNet 靠攏

---

# 8. **最後提交 package（5 分鐘）**

你需要交：

```
technical_summary.pdf
GitHub repo link
Optional: short Loom video walkthrough（會讓你大幅加分）
你的 available time for follow-up meeting
```

---

# 🚀 最終成果（面試官會超滿意）

你會交出：

* 一份專業的 1–2 頁 summary
* 一個乾淨、模組化的 ML pipeline
* 清楚 justify 的架構選擇
* 報告裡能展示你理解 MMIDNet，但又能簡化成可實作的版本
* 三種模型的比較（MLP / CNN / PointNet）

這會讓你在眾多 candidate 裡脫穎而出。

