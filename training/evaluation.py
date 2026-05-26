"""
Funciones para evaluar y guardar resultados de los modelos.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from preprocessing import process_audio_file


def evaluate_model(model, x_test, y_test, class_names):
    """
    Calcula metricas principales para un modelo entrenado.

    Retorna un diccionario simple para poder guardarlo como JSON.
    """
    probabilities = model.predict(x_test)
    y_pred = np.argmax(probabilities, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def evaluate_model_on_files(
    model,
    file_paths,
    labels,
    label_encoder,
    feature_type,
    segment_duration,
    overlap,
):
    """
    Evalua por archivo completo.

    Como ahora cada audio se divide en segmentos, primero se predice cada
    segmento y luego se promedian las probabilidades. Esta evaluacion se parece
    mas al uso real de la API.
    """
    y_true = []
    y_pred = []

    for file_path, label in zip(file_paths, labels):
        segments = process_audio_file(
            file_path,
            feature_type=feature_type,
            segment_duration=segment_duration,
            overlap=overlap,
        )
        segment_probabilities = model.predict(segments, verbose=0)
        probabilities = np.mean(segment_probabilities, axis=0)

        y_true.append(label_encoder.transform([label])[0])
        y_pred.append(int(np.argmax(probabilities)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_json(data, file_path):
    """Guarda un diccionario en formato JSON legible."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def plot_training_history(history, output_path, title):
    """Guarda graficas de accuracy y loss durante entrenamiento."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Entrenamiento")
    plt.plot(history.history["val_accuracy"], label="Validacion")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validacion")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(matrix, class_names, output_path, title):
    """Guarda una matriz de confusion como imagen."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Prediccion")
    plt.ylabel("Valor real")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
