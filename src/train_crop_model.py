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