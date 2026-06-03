# Crop Model 程式碼說明指南

## 專案概覽

這個專案有兩個 TensorFlow 作物推薦模型：

| 檔案 | 模型類型 | Accuracy |
|---|---|---|
| `src/train_crop_model_tf.py` | TensorFlow MLP v1 | ~0.9841 |
| `src/train_crop_model_tf_v2.py` | TensorFlow MLP v2（Ensemble）| ~0.9886 |

輸入：7 個土壤與環境特徵  
輸出：22 種作物中最適合種植的一種

---

## 資料集

`data/crop/Crop_recommendation.csv`

- 共 2200 筆資料，每種作物各 100 筆
- 7 個特徵欄：

| 欄位 | 說明 |
|------|-----|
| N | 土壤氮含量 |
| P | 土壤磷含量 |
| K | 土壤鉀含量 |
| temperature | 溫度（°C） |
| humidity | 濕度（%） |
| ph | 土壤酸鹼值 |
| rainfall | 雨量（mm）|

- 1 個標籤欄：`label`（作物名稱，文字）

---

## v1：`train_crop_model_tf.py`

### 流程

```
讀 CSV → 前處理（Poly + Scaler）→ 切三個集合 → 建神經網路 → 訓練（含 callback）→ 評估 → 存檔 → 單筆預測
```

### 各函式說明

#### `load_data()`
和同學版相同，讀 CSV。

#### `preprocess_data(df)`
共三個步驟：

```python
# 1. LabelEncoder：文字標籤 → 數字
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
# "rice" → 0, "maize" → 1, ... （共 22 種）

# 2. PolynomialFeatures：7 個原始特徵 → 35 個特徵
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# 原始 7 個 + 交叉項（N*P, N*K, ...）+ 平方項（N², P², ...）

# 3. StandardScaler：特徵縮放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)
# 每個特徵縮放成：(原始值 - 平均) / 標準差
```

**為什麼要加 PolynomialFeatures？**  
原始 7 個特徵只描述單一指標，交叉項（如 N×temperature）能讓模型學習到多個環境條件同時作用的效果，增加模型的表達能力。

**為什麼樹模型不需要 scaling，神經網路需要？**  
樹模型只看「哪個值比哪個大」，數值範圍不影響。  
神經網路用梯度下降更新參數，如果特徵數值差距太大（rainfall 最大 200，ph 最大 9），大數值的特徵會主導梯度，導致訓練不穩定。

**為什麼要把 scaler、encoder、poly 存起來？**  
預測新資料時必須用「同一個」前處理器，不能重新 fit（標準不一樣結果就會錯）。

#### `build_model(num_features, num_classes)`

網路結構（35 → 512 → 256 → 128 → 64 → 22）：

```
Input(35)
  ↓
Dense(512) + ReLU + L2(1e-4)   ← 學習複雜特徵組合
  ↓
BatchNormalization              ← 穩定每層數值，加速訓練
  ↓
Dropout(0.2)                   ← 訓練時隨機關掉 20% 神經元，防 overfitting
  ↓
Dense(256) + ReLU + L2(1e-4)
  ↓
BatchNormalization
  ↓
Dropout(0.15)
  ↓
Dense(128) + ReLU + L2(1e-4)
  ↓
BatchNormalization
  ↓
Dropout(0.1)
  ↓
Dense(64) + ReLU + L2(1e-4)
  ↓
Dense(22) + Softmax             ← 輸出 22 個作物的機率，加總為 1
```

**各元件說明：**
- `Dense`：全連接層，每個神經元都和上一層所有神經元相連
- `ReLU`：啟動函數，負數輸出 0、正數不變，讓模型學非線性關係
- `L2 regularization`：對較大的權重加罰分，避免模型對訓練資料過度擬合
- `BatchNormalization`：把每一批次的輸出標準化，讓訓練更穩定
- `Dropout`：防止模型過度依賴特定神經元（overfitting 的來源）
- `Softmax`：把 22 個數字轉成機率，選機率最高的當預測結果

**編譯設定：**
```python
optimizer = Adam(learning_rate=5e-4)
loss = "sparse_categorical_crossentropy"  # 多分類問題的標準 loss
metrics = ["accuracy"]
```

#### `train_model(...)`

```python
epochs=300      # 最多跑 300 輪
batch_size=16   # 每次用 16 筆資料更新一次參數
```

兩個 Callback（訓練控制器）：

| Callback | 作用 |
|---|---|
| `EarlyStopping(patience=30)` | val_accuracy 連續 30 epoch 沒進步就停，並還原最佳權重 |
| `ReduceLROnPlateau(patience=10)` | val_loss 連續 10 epoch 沒進步就把學習率乘以 0.5，最低降到 1e-6 |

#### `plot_history(history)`
畫訓練曲線並存到 `models/training_history.png`：
- 左圖：Train vs Val Accuracy（理想情況兩條線接近且都高）
- 右圖：Train vs Val Loss（理想情況兩條線接近且都低）

#### `evaluate_model(...)`
```python
model.evaluate(X_test, y_test)          # 輸出 loss 和 accuracy
np.argmax(model.predict(...), axis=1)   # 取機率最高的類別索引
classification_report(...)              # 各類別詳細報告
```

#### `save_artifacts(...)`
```python
model.save(TF_MODEL_PATH)         # 存模型（.keras 格式）
joblib.dump(scaler, SCALER_PATH)  # 存 scaler（.pkl）
joblib.dump(encoder, ENCODER_PATH)# 存 encoder（.pkl）
joblib.dump(poly, POLY_PATH)      # 存 poly（.pkl）
```

### 存檔
```
models/crop_tf_model.keras        ← 模型主體
models/crop_scaler.pkl            ← 特徵縮放器
models/crop_label_encoder.pkl     ← 標籤轉換器
models/crop_poly.pkl              ← 多項式展開器
models/training_history.png       ← 訓練曲線圖
```

### 資料切分方式

```
全部 2200 筆
  ├── 測試集（Test）：440 筆（20%）  ← 最後才用，模擬真實預測
  └── 訓練+驗證：1760 筆（80%）
        ├── 驗證集（Val）：176 筆（10%）← 監控訓練過程
        └── 訓練集（Train）：1584 筆（90%）← 實際訓練
```

`stratify=y` 確保每個集合裡 22 種作物的比例相同。

### 單筆預測流程

```python
sample_raw = np.array([[90, 42, 43, 20.87, 82.00, 6.5, 202.9]])
sample_poly = poly.transform(sample_raw)               # 套用同一個 poly 展開
sample_scaled = scaler.transform(sample_poly)          # 用同一個 scaler 縮放
pred_idx = np.argmax(model.predict(sample_scaled), axis=1)[0]  # 取最高機率的索引
pred_label = encoder.inverse_transform([pred_idx])[0]           # 數字轉回文字
# 輸出："rice"
```

---

## v2：`train_crop_model_tf_v2.py`（Ensemble）

v2 和 v1 的前處理、網路架構完全相同，針對 v1 的三個問題做改進。

### 改動 1：加入 GaussianNoise

**v1 的問題**：模型訓練時每次看到的輸入資料完全固定，容易死記訓練集的數值分布，導致對略有不同的測試資料表現下滑（overfitting 的一種形式）。

**v2 的做法**：在網路最前面加一層 GaussianNoise：
```python
layers.GaussianNoise(0.05)
```
訓練時對 35 個輸入特徵各加一個微小隨機值（標準差 0.05），**預測時自動關閉**，不影響推論。

**為什麼有效**：模型每個 epoch 看到的輸入都稍微不同，被迫學習特徵的「趨勢」而非精確數值，泛化能力更強。效果類似資料增強（data augmentation），但作用在特徵層而非原始資料。

---

### 改動 2：Cosine Annealing 取代 ReduceLROnPlateau

**v1 的做法**：用 `ReduceLROnPlateau`，當 val_loss 連續 10 epoch 沒進步才把學習率砍半。

**v1 的問題**：這是被動策略——只有卡住才反應，學習率只會單調下降，一旦降低就無法回升，容易困在局部最優解。

**v2 的改法**：改成 Cosine Annealing with Warm Restarts：
```python
CosineDecayRestarts(
    initial_learning_rate=5e-4,
    first_decay_steps=50,   # 第一個週期 50 epoch
    t_mul=2.0,              # 每個週期長度 ×2
    m_mul=0.9,              # 每個週期起始學習率 ×0.9
)
```

學習率主動按餘弦曲線下降，每個週期結束後重置（Warm Restart）：
```
學習率
5e-4 ┐         ┐
     │╲       /│╲
     │  ╲   /  │  ╲
     │    ╲/   │    ...
0    └─────────└──────→ epoch
     0   50   150
```

**為什麼有效**：學習率高時步伐大、能探索；下降時步伐小、能精確收斂。每次 Restart 讓模型有機會從局部最優解跳出，再找更好的解。相比 ReduceLROnPlateau 的單調下降，這種週期性變化讓訓練更有機會找到全域最優。

---

### 改動 3：Ensemble（核心改動）

**v1 的問題**：單一模型的預測結果受初始隨機權重影響，訓練結果有一定的隨機性，某些邊界案例可能預測不穩定。

**v2 的做法**：用 5 個不同 seed 各訓練一個完整模型，預測時取平均機率：
```python
seeds = [42, 7, 123, 256, 999]

# 預測時
probs = np.mean([m.predict(X) for m in models], axis=0)
y_pred = np.argmax(probs, axis=1)
```

**為什麼有效**：每個 seed 產生不同的初始權重，5 個模型雖然架構相同，但學到的決策邊界略有差異。對同一筆資料的預測：

```
Model 1: rice=0.91, wheat=0.06, ...
Model 2: rice=0.88, wheat=0.09, ...
Model 3: rice=0.94, wheat=0.04, ...
Model 4: rice=0.89, wheat=0.08, ...
Model 5: rice=0.92, wheat=0.05, ...
平均:    rice=0.908  ← 取這個當最終答案
```

單一模型的偶發錯誤會被其他 4 個模型的正確判斷壓過去，整體預測更穩定，這就是 0.9841 → 0.9886 的來源。代價是訓練時間和模型儲存空間都變成 5 倍。

### 存檔

```
models/crop_tf_v2_model_0.keras   ← 5 個模型各自存一個檔
models/crop_tf_v2_model_1.keras
models/crop_tf_v2_model_2.keras
models/crop_tf_v2_model_3.keras
models/crop_tf_v2_model_4.keras
models/crop_v2_scaler.pkl
models/crop_v2_encoder.pkl
models/crop_v2_poly.pkl
```

---

## 各版本進化對照

| 比較項目 | v1 | v2 |
|---|---|---|
| 特徵數 | 35（PolynomialFeatures）| 35 |
| 網路寬度 | 512→256→128→64 | 512→256→128→64 |
| L2 regularization | 1e-4 | 1e-4 |
| Dropout | 0.2/0.15/0.1 | 0.2/0.15/0.1 |
| GaussianNoise | 無 | 0.05 |
| 學習率排程 | ReduceLROnPlateau | Cosine Annealing |
| 模型數量 | 1 | 5（Ensemble）|
| Accuracy | ~0.9841 | ~0.9886 |

---

## 路徑說明

兩個檔案都用這個方式定位路徑，不管從哪裡執行都正確：

```python
BASE_DIR = Path(__file__).resolve().parent.parent
# __file__ → src/train_crop_model_tf.py
# .parent  → src/
# .parent  → crop/（根目錄）
```
