import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "data/crop/Crop_recommendation.csv"
MODEL_PATH = "models/crop_model.pkl"


def main():
    os.makedirs("models", exist_ok=True)

    # 1. 讀取資料
    df = pd.read_csv(DATA_PATH)

    print("資料前 5 筆：")
    print(df.head())

    print("\n欄位名稱：")
    print(df.columns)

    # 2. 切 X / y
    X = df.drop("label", axis=1)
    y = df["label"]

    # 3. 切訓練集 / 測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 4. 建立基本模型
    model = DecisionTreeClassifier(random_state=42)

    # 5. 訓練
    model.fit(X_train, y_train)

    # 6. 預測
    y_pred = model.predict(X_test)

    # 7. 評估
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 8. 儲存模型
    joblib.dump(model, MODEL_PATH)
    print(f"\n模型已儲存到：{MODEL_PATH}")

    # 9. 測試單筆預測
    sample = pd.DataFrame([{
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.87,
        "humidity": 82.00,
        "ph": 6.5,
        "rainfall": 202.9
    }])

    result = model.predict(sample)
    print("\n單筆測試預測結果：", result[0])


if __name__ == "__main__":
    main()