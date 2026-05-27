# %% [Step 4 — Naive Bayes：最簡單的假設，撞上天花板]
#
# Step 3 確認：線性邊界不夠，決策邊界是非線性的
#
# NB 的思路完全不同：不找邊界，直接問
#   「這筆資料的特徵值，在哪個作物的統計分佈下最合理？」
#
# Gaussian NB 的兩個假設：
#   1. 特徵在給定類別下條件獨立（Naive）
#   2. 每個特徵在每個類別下服從高斯分佈（Gaussian）
#
# 問題：P-K 相關係數 0.74，假設 1 明顯被違反
# 預期：NB 應該輸，但實際上...
#
# 學習目標：
#   1. 確認 NB 在原始 7 特徵下的成績（天花板？）
#   2. 可視化 class-conditional 分佈，理解為什麼假設違反了但結果還好
#   3. 特徵工程實驗：加 ratio 特徵 → NB 為什麼掉？
#   4. rice-jute 問題：NB 有沒有解決 LR 的痛點？

import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

DATA_PATH = r"D:\temppp\ML3\Crop_Recommendation_Plant_Disease_Detection-main\data\crop\Crop_recommendation.csv"

FIG_DIR = "models/step4_figs"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop("label", axis=1)
y = df["label"]
FEATURES = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% [實驗 1：NB 基本結果（原始 7 特徵）]

print("=" * 60)
print("Gaussian NB 基本結果（原始 7 特徵）")
print("=" * 60)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(nb, X_train, y_train, cv=cv, scoring="accuracy")

print(f"  Test Acc : {test_acc:.4f}")
print(f"  CV Mean  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  vs RF    : {'同一天花板' if test_acc >= 0.995 else f'差距 {0.9955 - test_acc:.4f}'}")
print(f"  vs LR    : +{test_acc - 0.9841:.4f}（比線性模型最佳成績好）")

# %% [實驗 2：為什麼 NB 能贏？— Class-conditional 分佈視覺化]
#
# NB 假設每個作物的每個特徵都是 Gaussian 分佈
# 如果各作物的 Gaussian 分佈在特徵空間中幾乎不重疊，
# 即使獨立假設被違反，NB 還是能正確分類

print("\n" + "=" * 60)
print("Class-conditional 分佈 — NB 的假設成不成立？")
print("=" * 60)

# 選幾個關鍵特徵，畫出各作物的分佈
key_features = ["rainfall", "humidity", "temperature", "ph"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, feat in zip(axes, key_features):
    for crop in sorted(y.unique()):
        vals = df[df["label"] == crop][feat]
        ax.plot(np.sort(vals),
                np.linspace(0, 1, len(vals)),
                alpha=0.5, linewidth=1)
    ax.set_title(f"{feat} — 各作物累積分佈")
    ax.set_xlabel(feat)
    ax.set_ylabel("CDF")

plt.suptitle("各作物的 Class-conditional 分佈（線越分開代表 NB 越容易區分）", y=1.01)
plt.tight_layout()
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.savefig(f"{FIG_DIR}/class_conditional_dist.png", dpi=120)
plt.close()
print(f"  [圖] 已儲存 {FIG_DIR}/class_conditional_dist.png")

# 計算各類別在每個特徵上的分佈重疊程度
print("\n  各作物在 rainfall 上的 mean ± std（看分離程度）：")
rainfall_stats = (df.groupby("label")["rainfall"]
                    .agg(["mean", "std"])
                    .sort_values("mean"))
for crop, row in rainfall_stats.iterrows():
    bar = "█" * int(row["mean"] / 10)
    print(f"    {crop:<15} {row['mean']:6.1f} ± {row['std']:5.1f}  {bar}")

# %% [實驗 3：NB 錯甚麼

print("\n" + "=" * 60)
print("=" * 60)

classes = sorted(y.unique())
cm_nb = confusion_matrix(y_test, y_pred, labels=classes)
cm_df = pd.DataFrame(cm_nb, index=classes, columns=classes)
plt.figure(figsize=(10, 8))
# NB 的全部錯誤
errors = []
for i, tc in enumerate(classes):
    for j, pc in enumerate(classes):
        if i != j and cm_nb[i, j] > 0:
            errors.append({"真實": tc, "預測錯成": pc, "次數": cm_nb[i, j]})

if errors:
    err_df = pd.DataFrame(errors).sort_values("次數", ascending=False)
    print(f"\n  NB 的錯誤配對（共 {sum(e['次數'] for e in errors)} 個錯誤）：")
    for _, row in err_df.iterrows():
        print(f"    {row['真實']:<15} → {row['預測錯成']:<15} {int(row['次數'])} 次")
else:
    print("\n  NB 零錯誤！Test Acc = 1.0000")

# %% [實驗 4：特徵工程對 NB 的影響（核心故事）]
#
# 加入 ratio 特徵：這些特徵對 tree-based 模型是額外資訊
# 但對 NB 是毒藥——因為 ratio 特徵分佈嚴重右偏，不是 Gaussian

print("\n" + "=" * 60)
print("特徵工程 — NB 的致命弱點")
print("=" * 60)

df_eng = df.copy()
df_eng["N_P_ratio"]             = df_eng["N"] / (df_eng["P"] + 1e-6)
df_eng["N_K_ratio"]             = df_eng["N"] / (df_eng["K"] + 1e-6)
df_eng["P_K_ratio"]             = df_eng["P"] / (df_eng["K"] + 1e-6)
df_eng["NPK_sum"]               = df_eng["N"] + df_eng["P"] + df_eng["K"]
df_eng["temp_humidity"]         = df_eng["temperature"] * df_eng["humidity"]
df_eng["rainfall_humidity_ratio"] = df_eng["rainfall"] / (df_eng["humidity"] + 1e-6)

ENG_FEATURES = [c for c in df_eng.columns if c != "label"]

X_eng = df_eng[ENG_FEATURES]
X_tr_eng, X_te_eng, y_tr, y_te = train_test_split(
    X_eng, df_eng["label"], test_size=0.2, random_state=42, stratify=df_eng["label"]
)

feature_sets = {
    "原始 7 特徵":    (X_train,   X_test),
    "全部 13 特徵":   (X_tr_eng,  X_te_eng),
}

print(f"  {'特徵組合':<16} {'NB Acc':>8}  {'RF Acc':>8}  NB 變化")
baseline_nb = test_acc
for fs_name, (X_tr, X_te) in feature_sets.items():
    nb_fs = GaussianNB()
    rf_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    nb_fs.fit(X_tr, y_train if X_tr is X_train else y_tr)
    rf_fs.fit(X_tr, y_train if X_tr is X_train else y_tr)
    y_true = y_test if X_tr is X_train else y_te
    nb_acc = accuracy_score(y_true, nb_fs.predict(X_te))
    rf_acc = accuracy_score(y_true, rf_fs.predict(X_te))
    change = nb_acc - baseline_nb
    sign = "+" if change >= 0 else ""
    print(f"  {fs_name:<16} {nb_acc:>8.4f}  {rf_acc:>8.4f}  {sign}{change:.4f}"
          + (" ← 下降！" if change < 0 else " ← 不變" if change == 0 else ""))

# 展示 ratio 特徵的分佈（為什麼它破壞 Gaussian 假設）
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
ratio_feats = ["N_P_ratio", "N_K_ratio", "P_K_ratio"]
for ax, feat in zip(axes, ratio_feats):
    df_eng[feat].hist(bins=50, ax=ax, color="tomato", edgecolor="white", alpha=0.8)
    ax.set_title(f"{feat}\n（右偏，非 Gaussian）")
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")

plt.suptitle("Ratio 特徵的分佈：右偏，違反 GaussianNB 的假設")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/ratio_feature_dist.png", dpi=120)
plt.close()
print(f"\n  [圖] 已儲存 {FIG_DIR}/ratio_feature_dist.png")

# %% [實驗 5：Permutation Importance — NB 靠哪個特徵？]

print("\n" + "=" * 60)
print("Permutation Importance — NB 最依賴哪個特徵？")
print("=" * 60)

pi = permutation_importance(nb, X_test, y_test, n_repeats=20, random_state=42)
pi_df = (pd.DataFrame({"feature": FEATURES,
                        "importance": pi.importances_mean,
                        "std": pi.importances_std})
           .sort_values("importance", ascending=False))

print("  打亂哪個特徵讓 NB 掉最多分：")
for _, row in pi_df.iterrows():
    bar = "█" * max(0, int(row["importance"] * 200))
    tag = " ← 拖累（加進去反而更差）" if row["importance"] < 0 else ""
    print(f"    {row['feature']:<15} {row['importance']:+.4f} ± {row['std']:.4f}  {bar}{tag}")

