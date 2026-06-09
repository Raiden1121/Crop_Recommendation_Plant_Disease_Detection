# %% [Step 3 — Logistic Regression：先問線性夠不夠]
#
# Step 2 的天花板：RF 和 ET 都到了 0.9955
# 在接受「需要複雜模型」之前，先問最簡單的問題：
#   如果用線性邊界，能切出 22 種作物嗎？
#
# 學習目標：
#   1. 用 LR 確認這份資料是非線性問題
#   2. 看哪些作物最難用線性分開（confusion matrix 分析）
#   3. 理解 LR 的限制在哪 → 為什麼後面的模型需要非線性能力

import os
import joblib
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_PATH = r"D:\temppp\ML3\Crop_Recommendation_Plant_Disease_Detection-main\data\crop\Crop_recommendation.csv"

FIG_DIR = "models/step3_figs"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# LR 對特徵尺度敏感，必須加 StandardScaler
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(max_iter=2000, random_state=42)),
])

# %% [實驗 1：LR 基本結果 vs RF 天花板]

print("=" * 60)
print("Logistic Regression 基本結果")
print("=" * 60)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring="accuracy")

print(f"  Test Acc : {test_acc:.4f}  (RF 天花板 0.9955，差距 {0.9955 - test_acc:.4f})")
print(f"  CV Mean  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# %% [實驗 2：C 參數（正則化強度）的影響]
#
# C = 1/lambda，C 越大正則化越弱（模型越複雜）
# 如果調大 C 能顯著提升，代表欠擬合；如果沒幫助，代表線性本身就是瓶頸

print("\n" + "=" * 60)
print("C 參數（正則化強度）的影響")
print("=" * 60)

c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
c_rows = []
for c in c_values:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=c, max_iter=2000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    t_acc = accuracy_score(y_test, pipe.predict(X_test))
    cv_m = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy").mean()
    c_rows.append({"C": c, "Test Acc": t_acc, "CV Mean": cv_m})
    print(f"  C={c:<8} Test={t_acc:.4f}  CV={cv_m:.4f}")

c_df = pd.DataFrame(c_rows)

fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogx(c_df["C"], c_df["Test Acc"], marker="o", label="Test Acc", color="steelblue")
ax.semilogx(c_df["C"], c_df["CV Mean"], marker="s", label="CV Mean",  color="tomato", linestyle="--")
ax.axhline(0.9955, color="gray", linestyle=":", alpha=0.7, label="RF 天花板 0.9955")
ax.set_xlabel("C（正則化強度的倒數，越大越複雜）")
ax.set_ylabel("Accuracy")
ax.set_title("Logistic Regression：C 值 vs Accuracy")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/lr_C_curve.png", dpi=120)
plt.close()
print(f"  [圖] 已儲存 {FIG_DIR}/lr_C_curve.png")

# %% [儲存最佳 LR 模型]
best_c = max(c_rows, key=lambda r: r["Test Acc"])["C"]
best_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(C=best_c, max_iter=2000, random_state=42)),
])
best_lr.fit(X_train, y_train)
joblib.dump(best_lr, "models/lr_best.pkl")
print(f"\n最佳 LR (C={best_c}) 已儲存到：models/lr_best.pkl")
