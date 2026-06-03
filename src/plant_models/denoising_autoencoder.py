import json
import math
import pickle
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ============================================================
# Basic settings
# ============================================================
SEED = 42
EPOCHS = 30
AUTOENCODER_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
INIT_LR = 1e-3
FINE_TUNE_LR = 1e-5
BS = 16
NOISE_FACTOR = 0.20
FREEZE_ENCODER_FIRST = True

# If this file is placed inside skeleton/ or scripts/, parents[1] is usually the project root.
# Example project structure:
# project/
#   data/plant_disease/PlantVillage/
#   models/
#   skeleton/this_file.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "plant_disease" / "PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models"
PLANT_MODELS_DIR = MODELS_DIR / "plant"

PLANT_MODEL_NAME = "denoising_autoencoder_classifier"
PLANT_MODEL_PATH = PLANT_MODELS_DIR / f"{PLANT_MODEL_NAME}.keras"
BEST_MODEL_PATH = PLANT_MODELS_DIR / f"{PLANT_MODEL_NAME}_best.keras"
AUTOENCODER_MODEL_PATH = PLANT_MODELS_DIR / "denoising_autoencoder.keras"
ENCODER_MODEL_PATH = PLANT_MODELS_DIR / "denoising_encoder.keras"

width = 256
height = 256
depth = 3
default_image_size = (height, width)
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# Device information
# ============================================================
def print_training_device():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("[INFO] Training device: GPU")
        for index, gpu in enumerate(gpus):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Memory growth must be set before GPUs are initialized.
                pass

            details = tf.config.experimental.get_device_details(gpu)
            device_name = details.get("device_name", gpu.name)
            compute_capability = details.get("compute_capability")
            capability_text = (
                f", compute capability: {compute_capability}"
                if compute_capability
                else ""
            )
            print(f"[INFO] GPU {index}: {device_name}{capability_text}")
    else:
        cpus = tf.config.list_physical_devices("CPU")
        cpu_names = ", ".join(cpu.name for cpu in cpus) or "CPU"
        print(f"[INFO] Training device: CPU ({cpu_names})")


# ============================================================
# Dataset collection / EDA
# ============================================================
def collect_image_paths():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}\n"
            "Please check your folder path. Expected structure:\n"
            "data/plant_disease/PlantVillage/class_name/image.jpg"
        )

    image_paths, label_list = [], []
    class_to_paths = {}

    print("[INFO] Collecting image paths ...")
    class_folders = sorted(
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir() and folder.name != ".DS_Store"
    )

    if not class_folders:
        raise ValueError(f"No class folders found in: {DATASET_DIR}")

    for class_folder in class_folders:
        class_image_paths = sorted(
            image_path
            for image_path in class_folder.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not class_image_paths:
            print(f"[WARNING] No images found in class folder: {class_folder.name}")
            continue

        print(f"[INFO] Processing {class_folder.name}: {len(class_image_paths)} images")
        class_to_paths[class_folder.name] = class_image_paths

        for image_path in class_image_paths:
            image_paths.append(str(image_path))
            label_list.append(class_folder.name)

    if not image_paths:
        raise ValueError(
            f"No images found in {DATASET_DIR}. Supported extensions: {IMAGE_EXTENSIONS}"
        )

    if len(set(label_list)) < 2:
        raise ValueError("At least 2 classes are required for classification.")

    return image_paths, label_list, class_to_paths


def show_dataset_summary(label_list, class_to_paths):
    class_counts = Counter(label_list)
    class_names = sorted(class_to_paths)

    print("\n[EDA] Dataset summary")
    print(f"[EDA] Total images: {len(label_list)}")
    print(f"[EDA] Total classes: {len(class_names)}")

    print("[EDA] Class names:")
    for class_name in class_names:
        print(f"  - {class_name}")

    print("[EDA] Images per class:")
    for class_name in class_names:
        print(f"  - {class_name}: {class_counts[class_name]}")

    counts = np.array(list(class_counts.values()))
    if np.any(counts < 3):
        print("[WARNING] Some classes have fewer than 3 images. Stratified splitting may fail.")

    median_count = np.median(counts)
    warning_threshold = max(1, int(median_count * 0.5))
    low_count_classes = [
        class_name
        for class_name, count in class_counts.items()
        if count < warning_threshold
    ]
    if low_count_classes:
        print(
            "[WARNING] Some classes have noticeably fewer images "
            f"than the median-based threshold ({warning_threshold}):"
        )
        for class_name in sorted(low_count_classes):
            print(f"  - {class_name}: {class_counts[class_name]}")


def plot_class_distribution(label_list):
    class_counts = Counter(label_list)
    class_names = sorted(class_counts)
    counts = [class_counts[class_name] for class_name in class_names]

    plt.figure(figsize=(14, 7))
    plt.bar(class_names, counts)
    plt.title("PlantVillage Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Image Count")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()

    output_path = PLANT_MODELS_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[EDA] Saved class distribution plot to {output_path}")


def show_sample_images(class_to_paths):
    class_names = sorted(class_to_paths)
    if not class_names:
        return

    columns = 5
    rows = math.ceil(len(class_names) / columns)
    plt.figure(figsize=(columns * 4, rows * 4))

    for index, class_name in enumerate(class_names, start=1):
        plt.subplot(rows, columns, index)
        sample_paths = class_to_paths[class_name]
        if sample_paths:
            image = plt.imread(sample_paths[0])
            plt.imshow(image)
        plt.title(class_name, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    output_path = PLANT_MODELS_DIR / "sample_images.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[EDA] Saved sample image grid to {output_path}")


# ============================================================
# Image loading / tf.data
# ============================================================
def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, default_image_size)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape((height, width, depth))
    return image, label


def add_noise(image):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=NOISE_FACTOR)
    noisy_image = image + noise
    noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
    return noisy_image


def build_classification_dataset(image_paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BS)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def build_autoencoder_dataset(image_paths, training=False):
    dummy_labels = np.zeros(len(image_paths), dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, dummy_labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def to_noisy_clean_pair(image_path, label):
        clean_image, _ = load_image(image_path, label)
        noisy_image = add_noise(clean_image)
        return noisy_image, clean_image

    dataset = dataset.map(to_noisy_clean_pair, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BS)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


# ============================================================
# Denoising Autoencoder model
# ============================================================
def build_denoising_autoencoder(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape, name="noisy_image_input")

    # Encoder: 256 -> 128 -> 64 -> 32
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same", name="encoded_features")(x)

    encoder = models.Model(inputs, encoded, name="denoising_encoder")

    # Decoder: 32 -> 64 -> 128 -> 256
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2D(
        3,
        (3, 3),
        padding="same",
        activation="sigmoid",
        name="reconstructed_image",
    )(x)

    autoencoder = models.Model(inputs, decoded, name="denoising_autoencoder")
    return autoencoder, encoder


def build_classifier_from_encoder(encoder, n_classes):
    inputs = layers.Input(shape=(height, width, depth), name="plant_image_input")

    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.08),
        ],
        name="data_augmentation",
    )

    x = augmentation(inputs)
    x = encoder(x)

    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(384, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(192, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="class_output")(x)

    model = models.Model(inputs, outputs, name="denoising_autoencoder_classifier")
    return model


# ============================================================
# Helpers
# ============================================================
def make_one_hot_labels(label_binarizer, label_array):
    labels = label_binarizer.transform(label_array).astype(np.float32)

    # LabelBinarizer returns shape (N, 1) for binary classification.
    # Categorical softmax output needs shape (N, 2), so fix it here.
    if len(label_binarizer.classes_) == 2 and labels.shape[1] == 1:
        labels = np.hstack([1.0 - labels, labels]).astype(np.float32)

    return labels


def build_callbacks(model_path, monitor="val_loss"):
    # val_accuracy 越高越好；val_loss 越低越好。
    mode = "max" if "accuracy" in monitor else "min"

    return [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def plot_classification_history(history, prefix="training"):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    if not acc or not val_acc or not loss or not val_loss:
        print("[WARNING] History does not contain accuracy/loss data, skip plotting.")
        return

    epochs = range(1, len(acc) + 1)
    accuracy_path = PLANT_MODELS_DIR / f"{prefix}_accuracy.png"
    loss_path = PLANT_MODELS_DIR / f"{prefix}_loss.png"

    plt.figure()
    plt.plot(epochs, acc, "b", label="Training accuracy")
    plt.plot(epochs, val_acc, "r", label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=150)
    plt.show()
    plt.close()
    print(f"[INFO] Saved accuracy curve to {accuracy_path}")

    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.show()
    plt.close()
    print(f"[INFO] Saved loss curve to {loss_path}")

def plot_autoencoder_history(history):
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(loss) + 1)

    output_path = PLANT_MODELS_DIR / "autoencoder_reconstruction_loss.png"

    plt.figure()
    plt.plot(epochs, loss, label="Training reconstruction loss")
    plt.plot(epochs, val_loss, label="Validation reconstruction loss")
    plt.title("Denoising Autoencoder Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()

    print(f"[INFO] Saved autoencoder loss curve to {output_path}")


def save_label_files(label_binarizer):
    with open(PLANT_MODELS_DIR / "label_transform.pkl", "wb") as label_file:
        pickle.dump(label_binarizer, label_file)

    with open(PLANT_MODELS_DIR / "class_names.json", "w", encoding="utf-8") as class_file:
        json.dump(label_binarizer.classes_.tolist(), class_file, indent=2, ensure_ascii=False)


def save_training_config(n_classes):
    config = {
        "model_name": PLANT_MODEL_NAME,
        "image_size": [height, width, depth],
        "batch_size": BS,
        "autoencoder_epochs": AUTOENCODER_EPOCHS,
        "classifier_epochs": EPOCHS,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "initial_learning_rate": INIT_LR,
        "fine_tune_learning_rate": FINE_TUNE_LR,
        "noise_factor": NOISE_FACTOR,
        "n_classes": n_classes,
        "dataset_dir": str(DATASET_DIR),
    }

    output_path = PLANT_MODELS_DIR / "training_config.json"
    with open(output_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved training config to {output_path}")


# ============================================================
# Main training process
# ============================================================
def main():
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLANT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print_training_device()

    image_paths, label_list, class_to_paths = collect_image_paths()
    show_dataset_summary(label_list, class_to_paths)
    plot_class_distribution(label_list)
    show_sample_images(class_to_paths)

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(label_list)
    save_label_files(label_binarizer)
    n_classes = len(label_binarizer.classes_)
    save_training_config(n_classes)

    image_paths = np.array(image_paths)
    image_labels = np.array(label_list)

    print("[INFO] Splitting data to train, validation, test")
    x_train_paths, x_temp_paths, y_train_labels, y_temp_labels = train_test_split(
        image_paths,
        image_labels,
        test_size=0.3,
        random_state=SEED,
        stratify=image_labels,
    )
    x_val_paths, x_test_paths, y_val_labels, y_test_labels = train_test_split(
        x_temp_paths,
        y_temp_labels,
        test_size=0.5,
        random_state=SEED,
        stratify=y_temp_labels,
    )

    y_train = make_one_hot_labels(label_binarizer, y_train_labels)
    y_val = make_one_hot_labels(label_binarizer, y_val_labels)
    y_test = make_one_hot_labels(label_binarizer, y_test_labels)

    train_ds = build_classification_dataset(x_train_paths, y_train, training=True)
    val_ds = build_classification_dataset(x_val_paths, y_val)
    test_ds = build_classification_dataset(x_test_paths, y_test)

    autoencoder_train_ds = build_autoencoder_dataset(x_train_paths, training=True)
    autoencoder_val_ds = build_autoencoder_dataset(x_val_paths)

    print(f"[INFO] Batch size: {BS}")
    print(
        "[INFO] Train/validation/test images: "
        f"{len(x_train_paths)}/{len(x_val_paths)}/{len(x_test_paths)}"
    )

    print("[INFO] Building denoising autoencoder...")
    autoencoder, encoder = build_denoising_autoencoder(
        input_shape=(height, width, depth)
    )
    autoencoder.summary()
    autoencoder.compile(
        loss="mse",
        optimizer=Adam(learning_rate=INIT_LR),
    )

    print("[INFO] Pretraining denoising autoencoder...")
    autoencoder_history = autoencoder.fit(
        autoencoder_train_ds,
        validation_data=autoencoder_val_ds,
        epochs=AUTOENCODER_EPOCHS,
        callbacks=build_callbacks(AUTOENCODER_MODEL_PATH, monitor="val_loss"),
        verbose=1,
    )
    plot_autoencoder_history(autoencoder_history)

    print(f"[INFO] Saving encoder to {ENCODER_MODEL_PATH}")
    encoder.save(ENCODER_MODEL_PATH)

    print("[INFO] Building classifier from pretrained encoder...")
    if FREEZE_ENCODER_FIRST:
        encoder.trainable = False
        print("[INFO] Encoder is frozen for the first classifier training stage.")

    model = build_classifier_from_encoder(encoder, n_classes)
    model.summary()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=INIT_LR),
        metrics=["accuracy"],
    )

    print("[INFO] Training classifier...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=build_callbacks(BEST_MODEL_PATH, monitor="val_accuracy"),
        verbose=1,
    )
    plot_classification_history(history, prefix="classifier")

    if FREEZE_ENCODER_FIRST and FINE_TUNE_EPOCHS > 0:
        print("[INFO] Fine-tuning classifier with encoder unfrozen...")
        encoder.trainable = True
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=FINE_TUNE_LR),
            metrics=["accuracy"],
        )
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=build_callbacks(BEST_MODEL_PATH, monitor="val_accuracy"),
            verbose=1,
        )
        plot_classification_history(fine_tune_history, prefix="fine_tune")

    print("[INFO] Calculating model accuracy")
    scores = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {scores[0]:.4f}")
    print(f"Test Accuracy: {scores[1] * 100:.2f}%")

    print(f"[INFO] Saving final classifier model to {PLANT_MODEL_PATH}")
    model.save(PLANT_MODEL_PATH)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
