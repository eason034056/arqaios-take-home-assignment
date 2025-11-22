# Quick Start Guide

這份指南幫助你快速開始使用這個專案。

## 代碼結構說明

### 1. 數據預處理 (`src/preprocessing.py`)

這個模組包含所有數據增強和預處理功能：

**核心函數解釋：**

- `normalize_to_unit_sphere(points)`: 
  - **字面意思**: "正規化到單位球體"
  - **作用**: 將點雲縮放到 [-1, 1] 範圍內
  - **為什麼**: 讓所有樣本有相同的尺度，方便模型學習
  - **使用方式**: `normalized = normalize_to_unit_sphere(points)`

- `random_rotation_z(points, angle_range=360)`:
  - **字面意思**: "沿 z 軸隨機旋轉"
  - **作用**: 隨機旋轉點雲 0-360 度
  - **為什麼**: 讓模型對人體朝向不敏感
  - **使用方式**: `rotated = random_rotation_z(points, angle_range=90)`

- `random_translation(points, translation_range=3.0)`:
  - **字面意思**: "隨機平移"
  - **作用**: 在 x-y 平面隨機移動點雲 ±3 公尺
  - **為什麼**: 讓模型對人體位置不敏感
  - **使用方式**: `translated = random_translation(points, translation_range=2.0)`

- `farthest_point_sampling(points, num_samples)`:
  - **字面意思**: "最遠點採樣"
  - **作用**: 選擇 N 個最分散的點
  - **為什麼**: 比隨機採樣更能保留整體形狀
  - **使用方式**: `sampled = farthest_point_sampling(points, 200)`

### 2. 數據集載入 (`src/dataset.py`)

**核心類別和函數：**

- `FAUSTPointCloudDataset`:
  - **字面意思**: "FAUST 點雲資料集"
  - **作用**: PyTorch Dataset 類別，用於批次載入數據
  - **使用方式**:
    ```python
    dataset = FAUSTPointCloudDataset(data, labels, augment=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    ```

- `load_faust_dataset(data_dir, num_points=200)`:
  - **字面意思**: "載入 FAUST 資料集"
  - **作用**: 從原始 mesh 檔案轉換成點雲
  - **為什麼**: FAUST 提供的是 mesh，需要轉成點雲格式
  - **使用方式**: `data, labels, files = load_faust_dataset("data/raw")`

- `stratified_split(data, labels, train_ratio=0.7, ...)`:
  - **字面意思**: "分層分割"
  - **作用**: 將數據分成 train/val/test，保持類別平衡
  - **為什麼**: 確保每個集合都有所有受試者的樣本
  - **使用方式**: 
    ```python
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(data, labels)
    ```

### 3. 模型架構

#### MLP Baseline (`src/models/mlp.py`)

- `MLPBaseline`:
  - **字面意思**: "多層感知機基線模型"
  - **架構**: Flatten → Dense → Dense → Output
  - **為什麼叫 Baseline**: 這是最簡單的模型，用來建立性能基準
  - **缺點**: 對點的順序敏感，無法理解幾何結構
  - **使用方式**:
    ```python
    model = MLPBaseline(num_points=200, num_classes=10)
    output = model(point_cloud)  # (B, 200, 3) → (B, 10)
    ```

#### 1D-CNN Model (`src/models/cnn1d.py`)

- `CNN1DModel`:
  - **字面意思**: "一維卷積神經網路模型"
  - **架構**: Sort → Conv1D → Conv1D → GlobalMaxPool → Dense
  - **為什麼用 1D**: 將點雲視為序列，用 1D 卷積捕捉局部模式
  - **改進**: 比 MLP 好，因為能學習局部特徵
  - **使用方式**:
    ```python
    model = CNN1DModel(num_points=200, num_classes=10)
    output = model(point_cloud)
    ```

#### Tiny PointNet (`src/models/pointnet_tiny.py`)

- `TinyPointNet`:
  - **字面意思**: "精簡版 PointNet"
  - **架構**: T-Net → Shared MLP → GlobalMaxPool → MLP
  - **為什麼最強**: 
    - T-Net 學習空間對齊（旋轉不變性）
    - Shared MLP 獨立處理每個點（排列不變性）
    - Global Max Pooling 聚合特徵（對稱函數）
  - **使用方式**:
    ```python
    model = TinyPointNet(num_points=200, num_classes=10, use_tnet=True)
    output = model(point_cloud)
    ```

- `TNet`:
  - **字面意思**: "變換網路"（Transformation Network）
  - **作用**: 預測一個 3×3 矩陣來對齊點雲
  - **為什麼需要**: 讓模型對旋轉和平移不敏感
  - **如何運作**: 
    ```python
    tnet = TNet(k=3)
    transform_matrix = tnet(point_cloud)  # 預測 3×3 矩陣
    aligned_pc = point_cloud @ transform_matrix  # 應用變換
    ```

### 4. 訓練腳本 (`src/train.py`)

**核心函數：**

- `train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)`:
  - **字面意思**: "訓練一個 epoch"
  - **作用**: 遍歷所有訓練數據一次，更新模型權重
  - **流程**:
    1. Forward pass: 計算預測
    2. Compute loss: 計算損失
    3. Backward pass: 計算梯度
    4. Update weights: 更新參數
  - **返回**: 平均損失和準確率

- `validate(model, val_loader, criterion, device)`:
  - **字面意思**: "驗證"
  - **作用**: 在驗證集上評估模型，不更新權重
  - **為什麼**: 檢查模型是否過擬合
  - **使用方式**: `val_loss, val_acc = validate(model, val_loader, criterion, device)`

- `EarlyStopping`:
  - **字面意思**: "提前停止"
  - **作用**: 如果驗證損失不再改善，停止訓練
  - **為什麼**: 防止過擬合，節省訓練時間
  - **使用方式**:
    ```python
    early_stopping = EarlyStopping(patience=10)
    if early_stopping(val_loss):
        print("停止訓練！")
        break
    ```

### 5. 評估腳本 (`src/evaluate.py`)

**核心函數：**

- `evaluate_model(model, test_loader, device)`:
  - **字面意思**: "評估模型"
  - **作用**: 在測試集上運行模型，收集預測結果
  - **返回**: y_true（真實標籤）, y_pred（預測標籤）, y_prob（預測機率）

- `plot_confusion_matrix(y_true, y_pred, class_names, save_path)`:
  - **字面意思**: "繪製混淆矩陣"
  - **作用**: 可視化哪些類別容易混淆
  - **解讀**: 
    - 對角線 = 正確預測
    - 非對角線 = 錯誤預測
  - **使用方式**: `plot_confusion_matrix(y_true, y_pred, class_names, "results/cm.png")`

- `compare_models(model_names, checkpoint_paths, ...)`:
  - **字面意思**: "比較模型"
  - **作用**: 同時評估多個模型，生成比較表格
  - **輸出**: CSV 檔案，包含準確率、F1、參數量等

## 快速開始步驟

### 步驟 1: 安裝環境

```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

### 步驟 2: 準備數據

```bash
# 下載 FAUST 數據集
# 訪問: http://faust.is.tue.mpg.de/
# 將 .ply 或 .obj 檔案放到 data/raw/
```

### 步驟 3: 訓練單一模型

```bash
# 訓練 PointNet（推薦）
python src/train.py --config config.yaml --model pointnet

# 或訓練 MLP（快速測試）
python src/train.py --config config.yaml --model mlp
```

### 步驟 4: 評估模型

```bash
# 評估單一模型
python src/evaluate.py \
    --model pointnet \
    --checkpoint results/checkpoints/pointnet/model_best.pth

# 比較所有模型
python src/evaluate.py --compare --models mlp cnn1d pointnet
```

### 步驟 5: 查看結果

```bash
# 啟動 TensorBoard 查看訓練曲線
tensorboard --logdir results/tensorboard

# 查看生成的圖表
ls results/
# - confusion_matrix_*.png
# - model_comparison.csv
# - per_class_metrics_*.png
```

## 常見問題

### Q1: 為什麼叫 "Tiny" PointNet？

**A**: 原始 PointNet 論文使用更深的網路和特徵變換。我們的版本簡化了架構，適合這個小規模任務（10 類分類），所以叫 "Tiny"。

### Q2: 為什麼使用 200 個點？

**A**: 
- mmWave 雷達通常輸出 <200 個點/幀
- 太少（<100）: 信息不足
- 太多（>500）: 計算成本高，且雷達做不到
- 200 是實用和性能的平衡

### Q3: Global Max Pooling 為什麼重要？

**A**: 
- **問題**: 點雲是無序集合，順序不應影響結果
- **解決**: Max pooling 是對稱函數：max([a,b,c]) = max([b,c,a])
- **結果**: 模型對點的順序不敏感（排列不變性）

### Q4: 為什麼 MLP 表現差？

**A**:
- MLP 將點雲展平成向量
- 順序改變 → 向量改變 → 預測改變
- 但點雲本質上是無序的，MLP 無法處理這個特性

### Q5: T-Net 做什麼？

**A**:
- **目的**: 學習一個變換矩陣來對齊點雲
- **類比**: 就像人看到物體會自動在腦中旋轉到標準視角
- **效果**: 讓模型對輸入的旋轉、平移不敏感

## 代碼閱讀建議

建議按以下順序閱讀代碼：

1. **preprocessing.py** - 理解數據如何處理
2. **dataset.py** - 理解數據如何載入
3. **models/mlp.py** - 從最簡單的模型開始
4. **models/cnn1d.py** - 看如何改進
5. **models/pointnet_tiny.py** - 理解最佳架構
6. **train.py** - 理解訓練流程
7. **evaluate.py** - 理解評估方法

每個檔案都有詳細的英文註解，解釋：
- 每個函數做什麼
- 為什麼這樣設計
- 如何使用
- 參數的意義

## 進階使用

### 修改超參數

編輯 `config.yaml`:

```yaml
training:
  batch_size: 64        # 增大 → 訓練更快但需要更多記憶體
  learning_rate: 0.0002 # 減小 → 訓練更穩定但更慢
  num_epochs: 120       # 增加 → 可能提高性能但可能過擬合
  
model:
  dropout: 0.3          # 增大 → 減少過擬合但可能欠擬合
```

### 使用 TensorBoard

```bash
# 訓練時會自動記錄
tensorboard --logdir results/tensorboard

# 可以看到：
# - 訓練/驗證損失曲線
# - 訓練/驗證準確率曲線
# - 學習率變化
```

### 繼續訓練

```bash
python src/train.py \
    --config config.yaml \
    --model pointnet \
    --resume results/checkpoints/pointnet/model_epoch_50.pth
```

## 總結

這個專案實現了三種點雲分類模型，展示了從簡單到複雜的進化：

1. **MLP**: 最簡單，性能差（~40%）
2. **1D-CNN**: 中等複雜度，性能中等（~70%）
3. **PointNet**: 最複雜，性能最佳（~80%）

每個模型都有完整的註解，解釋：
- 名稱的由來
- 設計的原因
- 使用的方法
- 預期的性能

希望這份指南能幫助你理解整個專案！

