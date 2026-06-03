# %% [Step 1 — Decision Tree 深入分析]
#
# 從 Step 0 我們知道 DT(0.9750) < RF(0.9909)
# 這一步要搞清楚：DT 輸在哪裡？
#
# 學習目標：
#   1. 理解 max_depth 如何造成 overfitting
#   2. 用 CV 量化 DT 的 variance 問題
#   3. 可視化樹的結構，看它到底在切什麼
#   4. 找出引入 RF 的具體理由

import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score
import numpy as np

FIG_DIR = "models/step1_figs"
os.makedirs(FIG_DIR, exist_ok=True)

DATA_PATH = r"D:\temppp\ML3\Crop_Recommendation_Plant_Disease_Detection-main\data\crop\Crop_recommendation.csv"

df = pd.read_csv(DATA_PATH)
X = df.drop("label", axis=1)
y = df["label"]
FEATURES = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% [實驗 1：max_depth 對 train/test acc 的影響]
#
# 假設：depth 越深 train acc 越高，但 test acc 先升後降
# 原因：DT 會把訓練集的噪聲和細節都記住（過擬合）

print("=" * 60)
print("max_depth vs Train / Test Accuracy")
print("=" * 60)

depths = list(range(1, 21)) + [None]
rows = []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc  = accuracy_score(y_test,  dt.predict(X_test))
    label = str(d) if d is not None else "None(全深)"
    rows.append({"max_depth": label, "Train Acc": train_acc, "Test Acc": test_acc})
    print(f"  depth={label:<10} train={train_acc:.4f}  test={test_acc:.4f}"
          + (" ← 過擬合開始" if train_acc - test_acc > 0.02 else ""))

depth_df = pd.DataFrame(rows)

# 畫圖
fig, ax = plt.subplots(figsize=(12, 5))
x_ticks = range(len(depth_df))
ax.plot(x_ticks, depth_df["Train Acc"], marker="o", label="Train Acc", color="steelblue")
ax.plot(x_ticks, depth_df["Test Acc"],  marker="s", label="Test Acc",  color="tomato")
ax.set_xticks(x_ticks)
ax.set_xticklabels(depth_df["max_depth"], rotation=45)
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_title("Decision Tree：max_depth vs Train/Test Accuracy")
ax.legend()
ax.set_ylim(0.7, 1.01)
ax.axhline(y=0.9750, color="gray", linestyle="--", alpha=0.5, label="Step0 基線")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/depth_vs_acc.png", dpi=120)
plt.close()
print(f"  [圖] 已儲存 {FIG_DIR}/depth_vs_acc.png")

# %% [實驗 2：Cross-Validation 量化 Variance]
#
# 用 CV std 衡量模型的不穩定程度
# DT 的 std 應該明顯高於 RF（下一步的預告）

print("\n" + "=" * 60)
print("Cross-Validation — 量化 DT 的 Variance")
print("=" * 60)

for d, label in [(3, "depth=3  (欠擬合)"),
                 (5, "depth=5  (平衡點附近)"),
                 (10, "depth=10 (過擬合)"),
                 (None, "depth=None (完全生長)")]:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  {label:<30} CV={scores.mean():.4f} ± {scores.std():.4f}  "
          f"(min={scores.min():.4f} max={scores.max():.4f})")
