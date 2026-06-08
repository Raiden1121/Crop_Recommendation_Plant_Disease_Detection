import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers, callbacks

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = str(BASE_DIR / "data" / "crop" / "Crop_recommendation.csv")
MODEL_DIR = str(BASE_DIR / "models")
TF_MODEL_PATH = str(BASE_DIR / "models" / "crop_tf_model.keras")
SCALER_PATH = str(BASE_DIR / "models" / "crop_scaler.pkl")
ENCODER_PATH = str(BASE_DIR / "models" / "crop_label_encoder.pkl")
POLY_PATH = str(BASE_DIR / "models" / "crop_poly.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df):
    X = df.drop("label", axis=1).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # 加入二次特徵：原始 7 個 → 原始 + 交叉項 + 平方項，共 35 個
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    return X_scaled, y, scaler, encoder, poly


def build_model(num_features, num_classes):
    reg = tf.keras.regularizers.l2(1e-4)

    model = tf.keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(512, activation="relu", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(256, activation="relu", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(128, activation="relu", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=16,
        callbacks=cb_list,
        verbose=1,
    )
    return history


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    import time
    run_id = time.strftime("%H%M%S")
    plt.savefig(f"{MODEL_DIR}/training_history_v1_{run_id}.png", dpi=150)
    print(f"Saved: training_history_v1_{run_id}.png")
    plt.show()


def evaluate_model(model, X_test, y_test, X_val, y_val, encoder):
    _, val_acc  = model.evaluate(X_val,  y_val,  verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nVal  Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    return test_acc


def save_artifacts(model, scaler, encoder, poly):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(TF_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(poly, POLY_PATH)
    print(f"\nModel saved to: {TF_MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Encoder saved to: {ENCODER_PATH}")
    print(f"Poly saved to: {POLY_PATH}")


def main():
    print("=== TensorFlow Crop Recommendation Model ===\n")

    df = load_data()
    X, y, scaler, encoder, poly = preprocess_data(df)
    num_classes = len(encoder.classes_)
    print(f"Classes ({num_classes}): {list(encoder.classes_)}")
    print(f"Features after polynomial expansion: {X.shape[1]}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    model = build_model(num_features=X.shape[1], num_classes=num_classes)
    model.summary()

    history = train_model(model, X_train, y_train, X_val, y_val)
    plot_history(history)

    acc = evaluate_model(model, X_test, y_test, X_val, y_val, encoder)

    save_artifacts(model, scaler, encoder, poly)

    # 單筆預測範例
    sample_raw = np.array([[90, 42, 43, 20.87, 82.00, 6.5, 202.9]])
    sample_poly = poly.transform(sample_raw)
    sample_scaled = scaler.transform(sample_poly)
    pred_idx = np.argmax(model.predict(sample_scaled, verbose=0), axis=1)[0]
    pred_label = encoder.inverse_transform([pred_idx])[0]
    print(f"\n單筆測試預測結果 (TensorFlow MLP): {pred_label}")


if __name__ == "__main__":
    main()
