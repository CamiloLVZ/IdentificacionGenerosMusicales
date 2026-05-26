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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import config
from augmentations import random_audio_augmentation, spec_augment
from evaluation import (
    evaluate_model_on_files,
    plot_confusion_matrix,
    plot_training_history,
)
from models import build_cnn_bilstm, build_simple_cnn
from preprocessing import (
    GENRES,
    RANDOM_SEED,
    audio_to_features,
    get_dataset_files,
    get_input_shape,
    load_audio,
    process_audio_file,
    split_audio_into_segments,
)
from utils import save_json, set_random_seed


def load_features_without_augmentation(file_paths, labels, feature_type, segment_duration, overlap):
    """
    Carga audios, los segmenta y genera features sin aumento de datos.

    Esta funcion se usa para validacion y prueba. Es importante no aumentar
    esos datos, porque queremos medir el rendimiento con audios reales.
    """
    x_data = []
    y_data = []

    for index, (file_path, label) in enumerate(zip(file_paths, labels), start=1):
        print(f"Procesando {index}/{len(file_paths)}: {Path(file_path).name}")
        features = process_audio_file(
            file_path,
            feature_type=feature_type,
            segment_duration=segment_duration,
            overlap=overlap,
        )

        for feature_matrix in features:
            x_data.append(feature_matrix)
            y_data.append(label)

    return np.array(x_data), np.array(y_data)


def load_training_features(
    file_paths,
    labels,
    feature_type,
    segment_duration,
    overlap,
    augmentations_per_segment,
):
    """
    Carga datos de entrenamiento con segmentacion y aumento de datos.

    Cada segmento original se conserva. Luego se crean algunas copias aumentadas
    al azar. Esto sube la cantidad de ejemplos sin volver enorme el proyecto.
    """
    x_data = []
    y_data = []

    for index, (file_path, label) in enumerate(zip(file_paths, labels), start=1):
        print(f"Segmentando y aumentando {index}/{len(file_paths)}: {Path(file_path).name}")
        audio, sr = load_audio(file_path)
        segments = split_audio_into_segments(audio, sr, segment_duration, overlap)

        for segment in segments:
            original_features = audio_to_features(segment, sr, feature_type, segment_duration)
            x_data.append(original_features)
            y_data.append(label)

            for _ in range(augmentations_per_segment):
                augmented_audio = random_audio_augmentation(segment, sr)
                augmented_features = audio_to_features(
                    augmented_audio,
                    sr,
                    feature_type,
                    segment_duration,
                )
                augmented_features = spec_augment(augmented_features)
                x_data.append(augmented_features)
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
    checkpoint_path = output_dir / "checkpoints" / f"{model_name}.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
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

    if checkpoint_path.exists():
        model = tf.keras.models.load_model(checkpoint_path)

    return model, history, best_val_accuracy


def run_cross_validation(args, audio_files, labels, label_encoder, output_dir):
    """
    Ejecuta validacion cruzada estratificada de 2 particiones.

    Se divide por archivo, no por segmento, para evitar que fragmentos de una
    misma cancion aparezcan en entrenamiento y validacion al mismo tiempo.
    """
    print("\nIniciando validacion cruzada estratificada de 2 folds")

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(audio_files, labels), start=1):
        print(f"\nFold {fold}/2")

        train_files = [audio_files[i] for i in train_index]
        val_files = [audio_files[i] for i in val_index]
        train_labels = [labels[i] for i in train_index]
        val_labels = [labels[i] for i in val_index]

        x_train, y_train_text = load_training_features(
            train_files,
            train_labels,
            args.feature_type,
            args.segment_duration,
            args.overlap,
            args.augmentations_per_segment,
        )
        x_val, y_val_text = load_features_without_augmentation(
            val_files,
            val_labels,
            args.feature_type,
            args.segment_duration,
            args.overlap,
        )

        y_train = label_encoder.transform(y_train_text)
        y_val = label_encoder.transform(y_val_text)

        model = build_cnn_bilstm(
            get_input_shape(args.feature_type, args.segment_duration),
            len(label_encoder.classes_),
        )

        model, history, best_val_accuracy = train_one_model(
            f"cross_validation_fold_{fold}_cnn_bilstm",
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            output_dir,
            args.cv_epochs,
            args.batch_size,
        )

        fold_accuracies.append(float(best_val_accuracy))
        plot_training_history(
            history,
            output_dir / "plots" / f"cross_validation_fold_{fold}_history.png",
            f"CV Fold {fold}",
        )

    cv_results = {
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
    }

    save_json(cv_results, output_dir / "metrics" / "cross_validation.json")
    return cv_results


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

    if not args.skip_cross_validation:
        cv_results = run_cross_validation(args, audio_files, labels, label_encoder, output_dir)
        print(
            "Validacion cruzada CNN+BiLSTM: "
            f"{cv_results['mean_accuracy']:.4f} +/- {cv_results['std_accuracy']:.4f}"
        )

    x_train, y_train_text_augmented = load_training_features(
        train_files,
        y_train_text,
        args.feature_type,
        args.segment_duration,
        args.overlap,
        args.augmentations_per_segment,
    )
    x_val, y_val_text = load_features_without_augmentation(
        validation_files,
        y_val_text,
        args.feature_type,
        args.segment_duration,
        args.overlap,
    )
    y_train = label_encoder.transform(y_train_text_augmented)
    y_val = label_encoder.transform(y_val_text)

    input_shape = get_input_shape(args.feature_type, args.segment_duration)
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

        metrics = evaluate_model_on_files(
            trained_model,
            test_files,
            y_test_text,
            label_encoder,
            args.feature_type,
            args.segment_duration,
            args.overlap,
        )
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

    preprocessing_config = {
        "feature_type": args.feature_type,
        "segment_duration": args.segment_duration,
        "overlap": args.overlap,
    }
    save_json(preprocessing_config, model_dir / "preprocessing_config.json")

    all_metrics["best_model"] = best_model_name
    save_json(all_metrics, output_dir / "metrics" / "metrics.json")

    print(f"Modelo guardado en: {model_dir / 'best_model.keras'}")
    print(f"Metricas guardadas en: {output_dir / 'metrics' / 'metrics.json'}")
    print(f"Configuracion de preprocesamiento guardada en: {model_dir / 'preprocessing_config.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelos con GTZAN")
    parser.add_argument("--dataset_path", default=config.DATASET_PATH, help="Ruta al dataset GTZAN")
    parser.add_argument("--model_dir", default=config.MODEL_DIR, help="Carpeta para guardar el modelo")
    parser.add_argument("--output_dir", default=config.OUTPUT_DIR, help="Carpeta para metricas y graficas")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Numero maximo de epocas")
    parser.add_argument("--cv_epochs", type=int, default=config.CV_EPOCHS, help="Epocas para cada fold de validacion cruzada")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Tamano del batch")
    parser.add_argument("--segment_duration", type=float, default=config.SEGMENT_DURATION, help="Duracion de cada segmento en segundos")
    parser.add_argument("--overlap", type=float, default=config.OVERLAP, help="Overlap entre segmentos. Ejemplo: 0.5")
    parser.add_argument(
        "--feature_type",
        default=config.FEATURE_TYPE,
        choices=["mel", "mfcc", "mfcc_delta"],
        help="Tipo de features usadas por el modelo",
    )
    parser.add_argument(
        "--augmentations_per_segment",
        type=int,
        default=config.AUGMENTATIONS_PER_SEGMENT,
        help="Copias aumentadas por cada segmento de entrenamiento",
    )
    parser.add_argument(
        "--skip_cross_validation",
        action="store_true",
        default=config.SKIP_CROSS_VALIDATION,
        help="Omite la validacion cruzada si se quiere una prueba mas rapida",
    )

    main(parser.parse_args())
