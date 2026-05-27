<<<<<<< HEAD
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/crop/Crop_recommendation.csv"
MODEL_DIR = "models"
DT_MODEL_PATH = f"{MODEL_DIR}/decision_tree.pkl"
RF_MODEL_PATH = f"{MODEL_DIR}/random_forest.pkl"

# Load data

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Exploratory Data Analysis (EDA)

def eda(df):
    print("資料前 5 筆：")
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe())

    print("\n欄位名稱：")
    print(df.columns)

    # 檢查label分布
    plt.figure(figsize=(10, 5))
    df["label"].value_counts().plot(kind="bar")
    plt.title("Crop Label Distribution")
    plt.xlabel("Crop Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 檢查feature correlation
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.drop("label", axis=1).corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation")
    plt.show()

    # boxplot
    for col in df.drop("label", axis=1).columns[:3]: # 只畫前3個特徵的boxplot
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot - {col}")
        plt.show()

# Preprocess data 

def preprocess_data(df):
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y

# Train model

def train_model(X_train, y_train):
    # Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)
    # Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    return dt_model, rf_model

# Evaluate model

def evaluate_model(dt_model, rf_model, X_test, y_test):
    # Evaluate Decision Tree model
    dt_pred = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    print("\nDecision Tree Accuracy:", dt_acc)
    print("\nDecision Tree Classification Report:")
    print(classification_report(y_test, dt_pred))

    # Evaluate Random Forest model
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print("\nRandom Forest Accuracy:", rf_acc)
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_pred))

    # 比較模型表現
    plt.bar(["Decision Tree", "Random Forest"], [dt_acc, rf_acc])
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    plt.show()

    return dt_acc, rf_acc

def save_models(dt_model, rf_model):

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(dt_model, DT_MODEL_PATH)
    joblib.dump(rf_model, RF_MODEL_PATH)

    print(f"\nModels saved to: {MODEL_DIR}/")

# =====================
# Main function 流程
# =====================
def main():

    # 1. 讀取資料
    df = load_data()
    
    # 2. EDA
    eda(df)

    # 3. 前處理資料
    x, y= preprocess_data(df)

    # 4. 切訓練集 / 測試集
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # 5. 訓練模型
    dt_model, rf_model = train_model(X_train, y_train)

    # 6. 評估模型
    dt_acc, rf_acc = evaluate_model(dt_model, rf_model, X_test, y_test)

    # 7. 儲存模型
    save_models(dt_model, rf_model)

    # 8. 測試單筆預測
    sample = pd.DataFrame([{
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.87,
        "humidity": 82.00,
        "ph": 6.5,
        "rainfall": 202.9
    }])

    result = rf_model.predict(sample)
    print("\n單筆測試預測結果(Random Forest)：", result[0])
# =====================
# 執行主程式
# =====================
if __name__ == "__main__":
    main()
=======
# %% [imports & 設定]
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, VotingClassifier, StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


class XGBWrapper(ClassifierMixin, BaseEstimator):
    """XGBoost wrapper：內部做 LabelEncoding，對外介面與 sklearn 一致。
    ClassifierMixin 必須在左邊，sklearn 1.8+ 的 MRO 才能正確設 estimator_type。
    __init__ 參數必須明確宣告，clone() 才能正確 introspect。
    """

    def __init__(self, n_estimators=150, random_state=42,
                 eval_metric="mlogloss", verbosity=0):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.verbosity = verbosity

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            eval_metric=self.eval_metric,
            verbosity=self.verbosity,
        )
        self._model.fit(X, y_enc)
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._model.predict(X))

    def predict_proba(self, X):
        return self._model.predict_proba(X)


_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC_DIR)
DATA_PATH = os.path.join(_ROOT, "data", "crop", "Crop_recommendation.csv")
MODEL_PATH = os.path.join(_ROOT, "models", "crop_model.pkl")
CV_FOLDS = 5


def build_models():
    models = {
        "Decision Tree":    DecisionTreeClassifier(random_state=42),
        "Random Forest":    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Extra Trees":      ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ]),
        "Naive Bayes": GaussianNB(),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)),
        ]),
    }
    try:
        import xgboost  # noqa: F401
        models["XGBoost"] = XGBWrapper(n_estimators=200, random_state=42,
                                        eval_metric="mlogloss", verbosity=0)
        print("[INFO] XGBoost 已載入")
    except ImportError:
        print("[INFO] XGBoost 未安裝，跳過（pip install xgboost）")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        print("[INFO] LightGBM 已載入")
    except ImportError:
        print("[INFO] LightGBM 未安裝，跳過（pip install lightgbm）")

    return models


def evaluate_all(models, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    rows, trained = [], {}
    for name, model in models.items():
        print(f"  訓練 {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="accuracy", n_jobs=-1)
        rows.append({"Model": name,
                     "Test Acc": round(test_acc, 4),
                     "CV Mean":  round(cv_scores.mean(), 4),
                     "CV Std":   round(cv_scores.std(), 4)})
        trained[name] = model
        print(f"Test={test_acc:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    df = pd.DataFrame(rows).sort_values("Test Acc", ascending=False).reset_index(drop=True)
    return df, trained


def show_feature_importance(trained, feature_names):
    for name in ("Random Forest", "Extra Trees", "XGBoost", "LightGBM"):
        if name not in trained:
            continue
        fi = pd.Series(trained[name].feature_importances_,
                       index=feature_names).sort_values(ascending=False)
        print(f"\n[特徵重要性 - {name}]")
        for feat, val in fi.items():
            print(f"  {feat:<15} {val:.4f}  {'█' * int(val * 40)}")
        break


# %% [Step 0：載入資料]
os.makedirs(os.path.join(_ROOT, "models"), exist_ok=True)

df = pd.read_csv(DATA_PATH)
FEATURES = [c for c in df.columns if c != "label"]
print(f"資料形狀：{df.shape}　特徵數：{len(FEATURES)}　類別數：{df['label'].nunique()}")
print(df.head())

# %% [Step 0.1：EDA 探索]
print("=== 缺失值 ===")
print(df.isnull().sum().to_string())

print(f"\n=== 重複列：{df.duplicated().sum()} 筆 ===")

print("\n=== 統計摘要 ===")
print(df[FEATURES].describe().round(2).to_string())

print("\n=== 類別分佈 ===")
print(df["label"].value_counts().to_string())

# 類別分佈長條圖
plt.figure(figsize=(14, 4))
df["label"].value_counts().plot(kind="bar", color="steelblue", edgecolor="white")
plt.title("類別分佈")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 特徵分佈直方圖
df[FEATURES].hist(bins=30, figsize=(14, 8), color="steelblue", edgecolor="white")
plt.suptitle("特徵分佈")
plt.tight_layout()
plt.show()

# 相關矩陣
plt.figure(figsize=(8, 6))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("特徵相關矩陣")
plt.tight_layout()
plt.show()

# 各作物平均特徵（找出最有區別力的特徵）
print("\n=== 各作物特徵平均值 ===")
print(df.groupby("label")[FEATURES].mean().round(1).to_string())

# %% [Step 0.2：資料清洗]
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"移除重複列：{before - len(df)} 筆，剩餘 {len(df)} 筆")

print("\n=== Outlier 偵測（IQR 法）===")
for feat in FEATURES:
    Q1, Q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[feat] < Q1 - 1.5 * IQR) | (df[feat] > Q3 + 1.5 * IQR)).sum()
    print(f"  {feat:<15} outlier 數：{n_out}")

# 農業資料的 outlier 通常是真實的極端氣候/土壤，不建議直接刪除
# 這裡只偵測不處理，記錄即可

# %% [Step 0.3：特徵工程]
df_feat = df[FEATURES].copy()

# 土壤養分比例（農學上有實際意義）
df_feat["N_P_ratio"]   = df_feat["N"] / (df_feat["P"] + 1e-6)
df_feat["N_K_ratio"]   = df_feat["N"] / (df_feat["K"] + 1e-6)
df_feat["P_K_ratio"]   = df_feat["P"] / (df_feat["K"] + 1e-6)
df_feat["NPK_sum"]     = df_feat["N"] + df_feat["P"] + df_feat["K"]

# 氣候交互項
df_feat["temp_humidity"]         = df_feat["temperature"] * df_feat["humidity"]
# jute vs rice 主要靠 rainfall 區分，ratio 加強信號
df_feat["rainfall_humidity_ratio"] = df_feat["rainfall"] / (df_feat["humidity"] + 1e-6)

X = df_feat
y = df["label"]
print(f"原始特徵：{len(FEATURES)} 個 → 工程後：{X.shape[1]} 個")
print(f"新增特徵：{[c for c in X.columns if c not in FEATURES]}")

# 工程後相關矩陣（看新特徵有沒有跟原始特徵高度重疊）
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, fmt=".1f", cmap="coolwarm", center=0,
            annot_kws={"size": 7})
plt.title("工程後特徵相關矩陣")
plt.tight_layout()
plt.show()

# %% [切訓練/測試集]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"訓練集：{X_train.shape}　測試集：{X_test.shape}")

# %% [Step 1：基礎模型比較]
print("\n" + "=" * 60)
print("STEP 1  基礎模型比較")
print("=" * 60)

models = build_models()
results_df, trained = evaluate_all(models, X_train, X_test, y_train, y_test)

print("\n排行榜：")
print(results_df.to_string(index=False))

show_feature_importance(trained, X.columns.tolist())

# %% [Step 1.5：特徵選擇分析]
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

print("=" * 60)
print("特徵選擇分析")
print("=" * 60)

# ── Permutation Importance（用 NB 和 RF 各跑一次）──────────────
for model_name, mdl in [("Naive Bayes", trained["Naive Bayes"]),
                         ("Random Forest", trained["Random Forest"])]:
    pi = permutation_importance(mdl, X_test, y_test,
                                n_repeats=20, random_state=42, n_jobs=-1)
    pi_df = (pd.DataFrame({"feature": X.columns,
                            "importance": pi.importances_mean,
                            "std": pi.importances_std})
               .sort_values("importance", ascending=False))
    print(f"\n[Permutation Importance — {model_name}]")
    for _, row in pi_df.iterrows():
        bar = "█" * max(0, int(row["importance"] * 100))
        tag = " ← 拖累" if row["importance"] < 0 else ""
        print(f"  {row['feature']:<28} {row['importance']:+.4f}  {bar}{tag}")

# ── 三種特徵組合比較 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("三種特徵組合 × 兩個模型 比較")
print("=" * 60)

feature_sets = {
    "7個（原始）":              FEATURES,
    "8個（原始 + rainfall_ratio）": FEATURES + ["rainfall_humidity_ratio"],
    "13個（全部工程）":          X.columns.tolist(),
}

comparison_rows = []
for fs_name, fs_cols in feature_sets.items():
    X_tr = X_train[fs_cols]
    X_te = X_test[fs_cols]
    for model_name, ModelClass, kwargs in [
        ("Naive Bayes",   GaussianNB,             {}),
        ("Random Forest", RandomForestClassifier, {"n_estimators": 200,
                                                    "random_state": 42,
                                                    "n_jobs": -1}),
    ]:
        m = ModelClass(**kwargs)
        m.fit(X_tr, y_train)
        acc = accuracy_score(y_test, m.predict(X_te))
        comparison_rows.append({"特徵組合": fs_name,
                                 "模型": model_name,
                                 "Test Acc": round(acc, 4)})

comp_df = pd.DataFrame(comparison_rows)
print(comp_df.pivot(index="特徵組合", columns="模型", values="Test Acc").to_string())

# %% [Step 2：Soft Voting]
print("\n" + "=" * 60)
print("STEP 2  Soft Voting（多樣性選法：每個模型家族各出一個最強）")
print("=" * 60)

MODEL_FAMILY = {
    "Decision Tree":     "tree",
    "Random Forest":     "tree",
    "Extra Trees":       "tree",
    "Gradient Boosting": "boosting",
    "XGBoost":           "boosting",
    "LightGBM":          "boosting",
    "SVM (RBF)":         "kernel",
    "KNN":               "distance",
    "Naive Bayes":       "probabilistic",
    "MLP":               "neural",
}

# 每個家族只取分數最高的那個，確保多樣性且不誤殺任何模型
top4 = (
    results_df
    .assign(family=lambda d: d["Model"].map(MODEL_FAMILY))
    .sort_values("Test Acc", ascending=False)
    .drop_duplicates("family")
    .head(4)["Model"]
    .tolist()
)
print(f"  參與模型：{top4}")

voting = VotingClassifier(
    estimators=[(n, trained[n]) for n in top4],
    voting="soft", n_jobs=-1,
)
voting.fit(X_train, y_train)
voting_acc = accuracy_score(y_test, voting.predict(X_test))
print(f"  Soft Voting Accuracy: {voting_acc:.4f}")

# %% [Step 3：Stacking]
print("\n" + "=" * 60)
print("STEP 3  Stacking（Top 4 + Logistic Regression meta）")
print("=" * 60)
print(f"  參與模型：{top4}")

stacking = StackingClassifier(
    estimators=[(n, trained[n]) for n in top4],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=CV_FOLDS, n_jobs=-1,
)
stacking.fit(X_train, y_train)
stacking_acc = accuracy_score(y_test, stacking.predict(X_test))
print(f"  Stacking Accuracy: {stacking_acc:.4f}")
print("\n  Classification Report（Stacking）：")
print(classification_report(y_test, stacking.predict(X_test)))

# %% [最終對比 & 儲存]
print("=" * 60)
print("最終對比")
print("=" * 60)
best_single = results_df.iloc[0]
summary = pd.DataFrame([
    {"方法": f"Best Single  ({best_single['Model']})", "Accuracy": best_single["Test Acc"]},
    {"方法": f"Soft Voting  ({', '.join(top4)})",      "Accuracy": round(voting_acc, 4)},
    {"方法": "Stacking     (meta=LR)",                 "Accuracy": round(stacking_acc, 4)},
])
print(summary.to_string(index=False))

best_acc = max(best_single["Test Acc"], voting_acc, stacking_acc)
if stacking_acc >= best_acc:
    best_model, best_label = stacking, "Stacking"
elif voting_acc >= best_acc:
    best_model, best_label = voting, "Soft Voting"
else:
    best_model = trained[best_single["Model"]]
    best_label = best_single["Model"]

joblib.dump(best_model, MODEL_PATH)
print(f"\n最佳模型（{best_label}, acc={best_acc:.4f}）已儲存：{MODEL_PATH}")

# %% [單筆測試]
sample = pd.DataFrame([{
    "N": 90, "P": 42, "K": 43,
    "temperature": 20.87, "humidity": 82.00,
    "ph": 6.5, "rainfall": 202.9,
}])
print(f"\n單筆測試預測結果：{best_model.predict(sample)[0]}")
>>>>>>> master
