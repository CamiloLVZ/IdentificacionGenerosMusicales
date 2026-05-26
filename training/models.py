"""
Arquitecturas de modelos para clasificacion de generos musicales.

Se implementan dos opciones:
1. CNN simple.
2. CNN + BiLSTM.

Las funciones devuelven modelos Keras listos para compilar y entrenar.
"""

from tensorflow.keras import layers, models, optimizers


def build_simple_cnn(input_shape, num_classes):
    """
    Crea una CNN pequena y facil de explicar.

    La CNN aprende patrones visuales del Mel Spectrogram, por ejemplo zonas con
    mucha energia en ciertas frecuencias.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.20),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.30),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.35),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="simple_cnn",
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_bilstm(input_shape, num_classes):
    """
    Crea un modelo CNN + BiLSTM.

    La CNN extrae caracteristicas del espectrograma. Luego la BiLSTM intenta
    aprender relaciones temporales, es decir, como cambian esas caracteristicas
    a lo largo del audio.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    # Convertimos la salida 2D de la CNN en una secuencia para la LSTM.
    # La dimension de tiempo queda como eje principal y las frecuencias/canales
    # se convierten en caracteristicas para cada paso temporal.
    x = layers.Reshape((x.shape[2], x.shape[1] * x.shape[3]))(x)

    x = layers.Bidirectional(layers.LSTM(96, dropout=0.20))(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_bilstm")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
