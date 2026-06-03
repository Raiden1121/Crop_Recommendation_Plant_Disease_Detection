import json
import math
import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam

# train model
import plant_models.transformer as plant_model


EPOCHS = 25
DEFAULT_INIT_LR = 1e-3
INIT_LR = getattr(plant_model, "INIT_LR", DEFAULT_INIT_LR)
BS = 16

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "plant_disease" / "PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models"
PLANT_MODELS_DIR = MODELS_DIR / "plant"
build_model = plant_model.build_model
PLANT_MODEL_NAME = build_model.__module__.rsplit(".", 1)[-1]
PLANT_MODEL_PATH = PLANT_MODELS_DIR / f"{PLANT_MODEL_NAME}.keras"

width = 256
height = 256
depth = 3
default_image_size = (height, width)
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def print_training_device():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("[INFO] Training device: GPU")
        for index, gpu in enumerate(gpus):
            tf.config.experimental.set_memory_growth(gpu, True)
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


def collect_image_paths():
    image_paths, label_list = [], []
    class_to_paths = {}

    print("[INFO] Collecting image paths ...")
    class_folders = sorted(
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir() and folder.name != ".DS_Store"
    )

    for class_folder in class_folders:
        print(f"[INFO] Processing {class_folder.name} ...")
        class_image_paths = sorted(
            image_path
            for image_path in class_folder.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        class_to_paths[class_folder.name] = class_image_paths

        for image_path in class_image_paths:
            image_paths.append(str(image_path))
            label_list.append(class_folder.name)

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

    if class_counts:
        counts = np.array(list(class_counts.values()))
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
    plt.show()
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
        plt.title(class_name, fontsize=16, wrap=True)
        plt.axis("off")

    plt.tight_layout()
    output_path = PLANT_MODELS_DIR / "sample_images.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()
    print(f"[EDA] Saved sample image grid to {output_path}")


def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, default_image_size)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape((height, width, depth))
    return image, label


def build_dataset(image_paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        dataset = dataset.shuffle(
            len(image_paths), seed=42, reshuffle_each_iteration=True
        )
    return (
        dataset
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .batch(BS)
        .prefetch(1)
    )


def plot_training_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    accuracy_path = PLANT_MODELS_DIR / "training_accuracy.png"
    loss_path = PLANT_MODELS_DIR / "training_loss.png"

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

    print(f"[INFO] Saved accuracy curve to {accuracy_path}")
    print(f"[INFO] Saved loss curve to {loss_path}")


def save_label_files(label_binarizer):
    with open(PLANT_MODELS_DIR / "label_transform.pkl", "wb") as label_file:
        pickle.dump(label_binarizer, label_file)

    with open(PLANT_MODELS_DIR / "class_names.json", "w", encoding="utf-8") as class_file:
        json.dump(label_binarizer.classes_.tolist(), class_file, indent=2)


def build_training_callbacks():
    callback_builder = getattr(plant_model, "build_callbacks", None)
    if callback_builder is None:
        return []

    try:
        return callback_builder(PLANT_MODELS_DIR)
    except TypeError:
        return callback_builder()


def main():
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

    image_paths = np.array(image_paths)
    image_labels = np.array(label_list)
    print("[INFO] Splitting data to train, validation, test")
    x_train_paths, x_temp_paths, y_train_labels, y_temp_labels = train_test_split(
        image_paths,
        image_labels,
        test_size=0.3,
        random_state=42,
        stratify=image_labels,
    )
    x_val_paths, x_test_paths, y_val_labels, y_test_labels = train_test_split(
        x_temp_paths,
        y_temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=y_temp_labels,
    )

    y_train = label_binarizer.transform(y_train_labels).astype(np.float32)
    y_val = label_binarizer.transform(y_val_labels).astype(np.float32)
    y_test = label_binarizer.transform(y_test_labels).astype(np.float32)

    train_ds = build_dataset(x_train_paths, y_train, training=True)
    val_ds = build_dataset(x_val_paths, y_val)
    test_ds = build_dataset(x_test_paths, y_test)

    print(f"[INFO] Batch size: {BS}")
    print(f"[INFO] Initial learning rate: {INIT_LR}")
    print(
        "[INFO] Train/validation/test images: "
        f"{len(x_train_paths)}/{len(x_val_paths)}/{len(x_test_paths)}"
    )

    model = build_model(n_classes, input_shape=(height, width, depth))
    model.summary()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=INIT_LR),
        metrics=["accuracy"],
    )

    print("[INFO] Training network...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=build_training_callbacks(),
        verbose=1,
    )

    plot_training_history(history)

    print("[INFO] Calculating model accuracy")
    scores = model.evaluate(test_ds)
    print(f"Test Accuracy: {scores[1] * 100}")

    print(f"[INFO] Saving model to {PLANT_MODEL_PATH}")
    model.save(PLANT_MODEL_PATH)


if __name__ == "__main__":
    main()
