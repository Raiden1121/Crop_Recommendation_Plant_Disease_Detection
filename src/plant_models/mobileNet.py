import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


height = 256
width = 256
depth = 3


def build_model(n_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(height, width, depth),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = Sequential()
    model.add(tf.keras.Input(shape=(height, width, depth)))

    model.add(tf.keras.layers.RandomFlip("horizontal"))
    model.add(tf.keras.layers.RandomRotation(0.07))
    model.add(tf.keras.layers.RandomZoom(0.1))

    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(Dense(n_classes, activation="softmax"))

    return model
