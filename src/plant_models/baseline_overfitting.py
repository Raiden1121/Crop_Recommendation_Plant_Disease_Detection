import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


def build_model(n_classes, input_shape=(256, 256, 3)):
    model = Sequential(name="baseline_overfitting_regularized")
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.RandomFlip("horizontal"))
    model.add(tf.keras.layers.RandomRotation(0.07))
    model.add(tf.keras.layers.RandomZoom(0.1))
    model.add(tf.keras.layers.RandomTranslation(0.1, 0.1))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation="softmax"))
    return model


def build_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
