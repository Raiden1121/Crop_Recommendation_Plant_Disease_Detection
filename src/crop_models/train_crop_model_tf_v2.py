import os
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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data" / "crop" / "Crop_recommendation.csv")
MODEL_DIR = str(BASE_DIR / "models")

N_MODELS = 5   # Ensemble 模型數量


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df):
    X = df.drop("label", axis=1).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    return X_scaled, y, scaler, encoder, poly


def build_model(num_features, num_classes, seed=42):
    tf.random.set_seed(seed)
    reg = tf.keras.regularizers.l2(1e-4)

    model = tf.keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.GaussianNoise(0.05),            # 訓練時加入微小雜訊，增強泛化
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

    # Cosine Annealing：學習率按餘弦曲線週期性下降
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=5e-4,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_single(model, X_train, y_train, X_val, y_val):
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            restore_best_weights=True,
            verbose=0,
        ),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=16,
        callbacks=cb_list,
        verbose=0,
    )
    return model


def train_ensemble(X_train, y_train, X_val, y_val, num_features, num_classes):
    models = []
    seeds = [42, 7, 123, 256, 999]

    for i, seed in enumerate(seeds):
        print(f"  Training model {i+1}/{N_MODELS} (seed={seed})...")
        model = build_model(num_features, num_classes, seed=seed)
        model = train_single(model, X_train, y_train, X_val, y_val)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"    Val Accuracy: {val_acc:.4f}")
        models.append(model)

    return models


def ensemble_predict(models, X):
    probs = np.mean([m.predict(X, verbose=0) for m in models], axis=0)
    return np.argmax(probs, axis=1)


def evaluate_ensemble(models, X_test, y_test, encoder):
    y_pred = ensemble_predict(models, X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nEnsemble Test Accuracy ({N_MODELS} models): {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    return acc


def save_artifacts(models, scaler, encoder, poly):
    os.makedirs(MODEL_DIR, exist_ok=True)
    for i, model in enumerate(models):
        path = str(BASE_DIR / "models" / f"crop_tf_v2_model_{i}.keras")
        model.save(path)
    joblib.dump(scaler, str(BASE_DIR / "models" / "crop_v2_scaler.pkl"))
    joblib.dump(encoder, str(BASE_DIR / "models" / "crop_v2_encoder.pkl"))
    joblib.dump(poly, str(BASE_DIR / "models" / "crop_v2_poly.pkl"))
    print(f"\nSaved {N_MODELS} models + scaler + encoder + poly to: {MODEL_DIR}/")


def main():
    print("=== TensorFlow Crop Recommendation Model v2 (Ensemble) ===\n")

    df = load_data()
    X, y, scaler, encoder, poly = preprocess_data(df)
    num_classes = len(encoder.classes_)
    print(f"Features after polynomial expansion: {X.shape[1]}")
    print(f"Classes: {num_classes}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    print(f"Training {N_MODELS} models for ensemble...\n")
    models = train_ensemble(X_train, y_train, X_val, y_val,
                            num_features=X.shape[1], num_classes=num_classes)

    acc = evaluate_ensemble(models, X_test, y_test, encoder)

    save_artifacts(models, scaler, encoder, poly)

    # 單筆預測範例
    sample_raw = np.array([[90, 42, 43, 20.87, 82.00, 6.5, 202.9]])
    sample_poly = poly.transform(sample_raw)
    sample_scaled = scaler.transform(sample_poly)
    pred_label = encoder.inverse_transform(
        ensemble_predict(models, sample_scaled)
    )[0]
    print(f"\n單筆測試預測結果 (Ensemble v2): {pred_label}")


if __name__ == "__main__":
    main()
