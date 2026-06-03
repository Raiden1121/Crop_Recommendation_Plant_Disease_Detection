# %% [Step 2 — Random Forest 深入：Bagging 降低 Variance]
# RF 的答案：訓練 N 棵各自帶隨機性的 DT，讓它們投票
# 隨機性來自兩個地方：
#   1. Bootstrap sampling：每棵樹只看隨機抽取（有放回）的子集資料
#   2. Feature subsampling：每次分裂只考慮隨機子集的特徵（sqrt(n_features)）
#
# 學習目標：
#   1. 量化 RF 比 DT 降低了多少 variance（CV std）
#   2. 觀察 n_estimators 對準確率的邊際效益
#   3. 讀懂 Feature Importance
#   4. Extra Trees vs RF：更極端的隨機化帶來什麼？
#   5. 找出引入 Boosting 的理由

import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = r"D:\temppp\ML3\Crop_Recommendation_Plant_Disease_Detection-main\data\crop\Crop_recommendation.csv"
FIG_DIR = "models/step2_figs"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop("label", axis=1)
y = df["label"]
FEATURES = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% [實驗 1：DT vs RF — Variance 直接比較]
#
# Bagging 的理論保證：如果各棵樹的誤差不相關，投票後 variance 下降到 1/N
# 實際上因為特徵有相關性，下降幅度小於 1/N，但仍然顯著

print("=" * 60)
print("：DT vs RF — Variance（CV Std）直接對比")
print("=" * 60)

dt_best = DecisionTreeClassifier(max_depth=10, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
results = {}
for name, model in [("DT (depth=10)", dt_best), ("RF (100棵)", rf)]:
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    results[name] = {"test": test_acc, "cv_mean": scores.mean(), "cv_std": scores.std()}
    print(f"  {name:<25} Test={test_acc:.4f}  CV={scores.mean():.4f} ± {scores.std():.4f}"
          + (f"  (std 降低 {(0.0068 - scores.std()) / 0.0068 * 100:.0f}%)" if "DT" not in name else ""))

# %% [實驗 2：n_estimators 邊際效益]
#
# 多少棵樹才「夠」？繼續加樹的邊際效益什麼時候遞減到幾乎沒有？

print("\n" + "=" * 60)
print("n_estimators 邊際效益（多少棵樹才夠？）")
print("=" * 60)

n_list = [1, 5, 10, 20, 50, 100, 200, 300, 500]
rf_rows = []
for n in n_list:
    rf_n = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf_n.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, rf_n.predict(X_test))
    cv_mean = cross_val_score(rf_n, X_train, y_train, cv=cv,
                              scoring="accuracy", n_jobs=-1).mean()
    rf_rows.append({"n_estimators": n, "Test Acc": test_acc, "CV Mean": cv_mean})
    print(f"  n={n:<5} Test={test_acc:.4f}  CV={cv_mean:.4f}")

rf_df = pd.DataFrame(rf_rows)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(rf_df["n_estimators"], rf_df["Test Acc"], marker="o", label="Test Acc", color="steelblue")
ax.plot(rf_df["n_estimators"], rf_df["CV Mean"], marker="s", label="CV Mean", color="tomato", linestyle="--")
ax.set_xlabel("n_estimators（樹的數量）")
ax.set_ylabel("Accuracy")
ax.set_title("RF：n_estimators vs Accuracy（邊際效益遞減）")
ax.legend()
ax.set_ylim(0.95, 1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/n_estimators.png", dpi=120)
plt.close()
print(f"  [圖] 已儲存 {FIG_DIR}/n_estimators.png")

