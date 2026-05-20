import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "../data/crop/Crop_recommendation.csv"
MODEL_PATH = "../models/crop_model.pkl"

def main():
    os.makedirs("../models", exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\n模型已儲存到：{MODEL_PATH}")

    sample = pd.DataFrame([{
        "N": 90, "P": 42, "K": 43,
        "temperature": 20.87, "humidity": 82.00,
        "ph": 6.5, "rainfall": 202.9
    }])
    print("\n單筆測試預測結果：", model.predict(sample)[0])


if __name__ == "__main__":
    main()