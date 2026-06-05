"""
train.py - Bangla Handwritten Character Recognition Training Script
Dataset: BanglaLekha-Isolated (84 classes)
Tracks experiments with MLflow, saves best model and label mapping.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.keras

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.10
RANDOM_SEED = 42
DATA_DIR = os.environ.get("DATA_DIR", "BanglaLekha-Isolated/Images")
MODEL_DIR = "models"
LABELS_PATH = "labels.json"
MLFLOW_EXPERIMENT = "BanglaOCR"


def load_dataset(data_dir: str, img_size: int = IMG_SIZE, max_per_class: int = 300):
    """
    Walk data_dir/{class_id}/*.png, resize to (img_size, img_size),
    normalise to [0,1]. Returns X, y arrays and label mapping dict.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found.\n"
            "Extract BanglaLekha-Isolated.zip next to train.py and set\n"
            "DATA_DIR=BanglaLekha-Isolated/Images or pass --data_dir."
        )

    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()],
                        key=lambda d: int(d.name))
    label_map = {d.name: idx for idx, d in enumerate(class_dirs)}

    X, y = [], []
    for class_dir in class_dirs:
        images = list(class_dir.glob("*.png"))[:max_per_class]
        for img_path in images:
            img = tf.keras.utils.load_img(
                img_path, color_mode="grayscale", target_size=(img_size, img_size)
            )
            arr = tf.keras.utils.img_to_array(img) / 255.0
            X.append(arr)
            y.append(label_map[class_dir.name])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Loaded {len(X)} images across {len(label_map)} classes.")
    return X, y, label_map


def build_cnn(num_classes: int, img_size: int = IMG_SIZE) -> tf.keras.Model:
    """Small custom CNN – fast to train, good baseline."""
    inp = layers.Input(shape=(img_size, img_size, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="BanglaCNN")


def build_deeper_cnn(num_classes: int, img_size: int = IMG_SIZE) -> tf.keras.Model:
    """Deeper variant used for second MLflow run."""
    inp = layers.Input(shape=(img_size, img_size, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="BanglaDeeperCNN")


ARCHITECTURES = {
    "cnn": build_cnn,
    "deeper_cnn": build_deeper_cnn,
}


def run_training(arch="cnn", learning_rate=LEARNING_RATE,
                 epochs=EPOCHS, batch_size=BATCH_SIZE,
                 data_dir=DATA_DIR, max_per_class=300):

    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, label_map = load_dataset(data_dir, max_per_class=max_per_class)
    num_classes = len(label_map)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
        random_state=RANDOM_SEED, stratify=y_trainval
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{arch}_lr{learning_rate}_ep{epochs}"):
        mlflow.log_params({
            "architecture": arch,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": IMG_SIZE,
            "num_classes": num_classes,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "max_per_class": max_per_class,
            "augmentation": True,
        })

        model = ARCHITECTURES[arch](num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        best_model_path = os.path.join(MODEL_DIR, "model.keras")

        cb_list = [
            callbacks.ModelCheckpoint(
                best_model_path, save_best_only=True,
                monitor="val_accuracy", verbose=1
            ),
            callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            ),
        ]

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=cb_list,
            verbose=1,
        )

        for epoch_idx, (acc, val_acc, loss, val_loss) in enumerate(zip(
            history.history["accuracy"],
            history.history["val_accuracy"],
            history.history["loss"],
            history.history["val_loss"],
        )):
            mlflow.log_metrics({
                "train_accuracy": acc,
                "val_accuracy": val_acc,
                "train_loss": loss,
                "val_loss": val_loss,
            }, step=epoch_idx)

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "best_val_accuracy": max(history.history["val_accuracy"]),
        })
        print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

        mlflow.keras.log_model(model, artifact_path="model")
        mlflow.log_artifact(best_model_path)

        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(LABELS_PATH)

        print(f"\nMLflow run_id: {mlflow.active_run().info.run_id}")
        print(f"Model saved to: {best_model_path}")
        print(f"Labels saved to: {LABELS_PATH}")

    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bangla OCR model")
    parser.add_argument("--arch", choices=list(ARCHITECTURES.keys()),
                        default="cnn")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--max_per_class", type=int, default=300,
                        help="Cap images per class (0 = all)")
    parser.add_argument("--run_both", action="store_true",
                        help="Run two MLflow experiments: cnn + deeper_cnn")
    args = parser.parse_args()

    max_pc = args.max_per_class if args.max_per_class > 0 else 99999

    if args.run_both:
        print("\n=== Run 1: cnn ===")
        run_training("cnn", args.lr, args.epochs, args.batch_size,
                     args.data_dir, max_pc)
        print("\n=== Run 2: deeper_cnn ===")
        run_training("deeper_cnn", args.lr * 0.5, args.epochs,
                     args.batch_size, args.data_dir, max_pc)
    else:
        run_training(args.arch, args.lr, args.epochs, args.batch_size,
                     args.data_dir, max_pc)
