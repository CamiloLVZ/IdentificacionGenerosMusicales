"""
Entrenamiento completo para clasificacion de generos musicales con GTZAN.

Uso recomendado desde la raiz del proyecto:
python training/train.py --dataset_path dataset

El script:
1. Lee audios WAV locales.
2. Genera Mel Spectrograms dinamicamente con librosa.
3. Divide datos en train/validation/test de forma estratificada.
4. Entrena dos modelos.
5. Compara resultados y guarda el mejor modelo.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
    save_json,
)
from models import build_cnn_bilstm, build_simple_cnn
from preprocessing import (
    EXPECTED_TIME_STEPS,
    GENRES,
    N_MELS,
    RANDOM_SEED,
    create_augmented_examples,
    get_dataset_files,
    process_audio_file,
)


def set_random_seed(seed):
    """Fija semillas para reducir variacion entre ejecuciones."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_spectrograms(file_paths, labels):
    """
    Carga audios y genera espectrogramas sin aumento de datos.

    Esta funcion se usa para validacion y prueba. Es importante no aumentar
    esos datos, porque queremos medir el rendimiento con audios reales.
    """
    x_data = []
    y_data = []

    for index, (file_path, label) in enumerate(zip(file_paths, labels), start=1):
        print(f"Procesando {index}/{len(file_paths)}: {Path(file_path).name}")
        spectrogram = process_audio_file(file_path)
        x_data.append(spectrogram)
        y_data.append(label)

    return np.array(x_data), np.array(y_data)


def load_training_spectrograms(file_paths, labels):
    """
    Carga datos de entrenamiento y aplica aumento de datos.

    Por cada audio se crean 4 ejemplos: original, ruido, shift temporal y pitch.
    Esto aumenta el conjunto de entrenamiento sin tocar validacion ni prueba.
    """
    x_data = []
    y_data = []

    for index, (file_path, label) in enumerate(zip(file_paths, labels), start=1):
        print(f"Aumentando {index}/{len(file_paths)}: {Path(file_path).name}")
        spectrograms = create_augmented_examples(file_path)

        for spectrogram in spectrograms:
            x_data.append(spectrogram)
            y_data.append(label)

    return np.array(x_data), np.array(y_data)


def split_dataset(audio_files, labels):
    """
    Divide el dataset en 70% entrenamiento, 15% validacion y 15% prueba.

    stratify mantiene una proporcion similar de generos en cada division.
    """
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        audio_files,
        labels,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    validation_files, test_files, validation_labels, test_labels = train_test_split(
        temp_files,
        temp_labels,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_labels,
    )

    return train_files, validation_files, test_files, train_labels, validation_labels, test_labels


def train_one_model(model_name, model, x_train, y_train, x_val, y_val, output_dir, epochs, batch_size):
    """Entrena un modelo y guarda la grafica de su historial."""
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        ),
    ]

    print(f"\nEntrenando modelo: {model_name}")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_history(
        history,
        output_dir / "plots" / f"{model_name}_history.png",
        model_name,
    )

    best_val_accuracy = max(history.history["val_accuracy"])
    return model, history, best_val_accuracy


def main(args):
    set_random_seed(RANDOM_SEED)

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    audio_files, labels = get_dataset_files(args.dataset_path)

    if len(audio_files) == 0:
        raise ValueError(
            "No se encontraron audios WAV. Revisa que dataset_path tenga carpetas por genero."
        )

    print(f"Audios encontrados: {len(audio_files)}")

    splits = split_dataset(audio_files, labels)
    train_files, validation_files, test_files, train_labels, validation_labels, test_labels = splits

    print(f"Train: {len(train_files)} audios")
    print(f"Validation: {len(validation_files)} audios")
    print(f"Test: {len(test_files)} audios")

    label_encoder = LabelEncoder()
    label_encoder.fit(GENRES)

    y_train_text = np.array(train_labels)
    y_val_text = np.array(validation_labels)
    y_test_text = np.array(test_labels)

    x_train, y_train_text_augmented = load_training_spectrograms(train_files, y_train_text)
    x_val, y_val_text = load_spectrograms(validation_files, y_val_text)
    x_test, y_test_text = load_spectrograms(test_files, y_test_text)

    y_train = label_encoder.transform(y_train_text_augmented)
    y_val = label_encoder.transform(y_val_text)
    y_test = label_encoder.transform(y_test_text)

    input_shape = (N_MELS, EXPECTED_TIME_STEPS, 1)
    num_classes = len(label_encoder.classes_)

    models_to_train = {
        "simple_cnn": build_simple_cnn(input_shape, num_classes),
        "cnn_bilstm": build_cnn_bilstm(input_shape, num_classes),
    }

    all_metrics = {}
    trained_models = {}
    validation_scores = {}

    for model_name, model in models_to_train.items():
        trained_model, history, val_accuracy = train_one_model(
            model_name,
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            output_dir,
            args.epochs,
            args.batch_size,
        )

        metrics = evaluate_model(trained_model, x_test, y_test, label_encoder.classes_)
        metrics["best_validation_accuracy"] = float(val_accuracy)

        all_metrics[model_name] = metrics
        trained_models[model_name] = trained_model
        validation_scores[model_name] = val_accuracy

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            label_encoder.classes_,
            output_dir / "plots" / f"{model_name}_confusion_matrix.png",
            f"{model_name} - Matriz de confusion",
        )

    best_model_name = max(validation_scores, key=validation_scores.get)
    best_model = trained_models[best_model_name]

    print(f"\nMejor modelo segun validation accuracy: {best_model_name}")

    best_model.save(model_dir / "best_model.keras")

    with open(model_dir / "label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)

    all_metrics["best_model"] = best_model_name
    save_json(all_metrics, output_dir / "metrics" / "metrics.json")

    print(f"Modelo guardado en: {model_dir / 'best_model.keras'}")
    print(f"Metricas guardadas en: {output_dir / 'metrics' / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelos con GTZAN")
    parser.add_argument("--dataset_path", default="dataset", help="Ruta al dataset GTZAN")
    parser.add_argument("--model_dir", default="models", help="Carpeta para guardar el modelo")
    parser.add_argument("--output_dir", default="outputs", help="Carpeta para metricas y graficas")
    parser.add_argument("--epochs", type=int, default=50, help="Numero maximo de epocas")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamano del batch")

    main(parser.parse_args())

