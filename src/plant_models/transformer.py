import tensorflow as tf
from tensorflow.keras import layers


INIT_LR = 1e-4


@tf.keras.utils.register_keras_serializable(package="plant_models")
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = tf.shape(patches)[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


@tf.keras.utils.register_keras_serializable(package="plant_models")
class ClassToken(layers.Layer):
    def __init__(self, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.projection_dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, encoded_patches):
        batch_size = tf.shape(encoded_patches)[0]
        return tf.tile(self.cls_token, [batch_size, 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({"projection_dim": self.projection_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="plant_models")
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.class_token = ClassToken(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1,
            output_dim=projection_dim,
        )

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.projection.build(input_shape)
        projected_shape = input_shape[:-1] + (self.projection_dim,)
        self.class_token.build(projected_shape)
        self.position_embedding.build((self.num_patches + 1,))
        super().build(input_shape)

    def call(self, patches):
        encoded_patches = self.projection(patches)
        cls_token = self.class_token(encoded_patches)
        encoded_tokens = tf.concat([cls_token, encoded_patches], axis=1)
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        return encoded_tokens + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "projection_dim": self.projection_dim,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="plant_models")
class TransformerEncoder(layers.Layer):
    def __init__(
        self,
        projection_dim,
        num_heads,
        mlp_dim,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=dropout_rate,
        )
        self.dropout_1 = layers.Dropout(dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp_dense_1 = layers.Dense(mlp_dim, activation=tf.nn.gelu)
        self.mlp_dropout_1 = layers.Dropout(dropout_rate)
        self.mlp_dense_2 = layers.Dense(projection_dim)
        self.mlp_dropout_2 = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.norm_1.build(input_shape)
        self.attention.build(input_shape, input_shape)
        self.dropout_1.build(input_shape)
        self.norm_2.build(input_shape)
        self.mlp_dense_1.build(input_shape)
        mlp_shape = input_shape[:-1] + (self.mlp_dim,)
        self.mlp_dropout_1.build(mlp_shape)
        self.mlp_dense_2.build(mlp_shape)
        self.mlp_dropout_2.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.norm_1(inputs)
        attention_output = self.attention(x, x, training=training)
        attention_output = self.dropout_1(attention_output, training=training)
        x = inputs + attention_output

        y = self.norm_2(x)
        y = self.mlp_dense_1(y)
        y = self.mlp_dropout_1(y, training=training)
        y = self.mlp_dense_2(y)
        y = self.mlp_dropout_2(y, training=training)
        return x + y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "projection_dim": self.projection_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="plant_models")
class ExtractClassToken(layers.Layer):
    def call(self, encoded_tokens):
        return encoded_tokens[:, 0]


def build_model(n_classes, input_shape=(256, 256, 3)):
    patch_size = 16
    projection_dim = 128
    transformer_layers = 6
    num_heads = 4
    transformer_mlp_dim = 256
    dropout_rate = 0.1
    mlp_head_units = [512, 256]

    if input_shape[0] % patch_size != 0 or input_shape[1] % patch_size != 0:
        raise ValueError("Input height and width must be divisible by patch_size.")

    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    inputs = layers.Input(shape=input_shape)

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.07)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    patches = Patches(patch_size=patch_size)(x)
    encoded_tokens = PatchEncoder(
        num_patches=num_patches,
        projection_dim=projection_dim,
    )(patches)

    for layer_index in range(transformer_layers):
        encoded_tokens = TransformerEncoder(
            projection_dim=projection_dim,
            num_heads=num_heads,
            mlp_dim=transformer_mlp_dim,
            dropout_rate=dropout_rate,
            name=f"transformer_encoder_{layer_index + 1}",
        )(encoded_tokens)

    encoded_tokens = layers.LayerNormalization(epsilon=1e-6)(encoded_tokens)
    cls_token = ExtractClassToken(name="cls_token_output")(encoded_tokens)
    x = layers.Dropout(0.5)(cls_token)

    for units in mlp_head_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="plant_vit")


def build_callbacks(model_dir=None):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if model_dir is not None:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_dir / "best_vit_plant_disease_model.keras",
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            )
        )

    return callbacks
