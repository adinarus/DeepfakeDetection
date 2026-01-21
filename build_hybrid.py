import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from classifiers import *

# =========================
# CONFIGURATION
# =========================

INPUT_SHAPE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 30

VAL_DATASET_DIR = r"C:\Users\Adi\Desktop\Licenta\datasets\ffpp\datasets_augmented\combined\valid"

BASE_MODELS = [
    {
        "name": "Meso4",
        "class": Meso4,
        "mode": "original",
        "path": r"data/saved_modelsMeso4.h5"
    },
    {
        "name": "Xception",
        "class": XceptionNet,
        "mode": "original",
        "path": r"data/saved_models/Xception.h5"
    },
    {
        "name": "F3Net",
        "class": F3NetClassifier,
        "mode": "FAD",
        "path": r"data/saved_models/F3NetFAD.h5"
    }
]

META_MODEL_OUTPUT = "data/saved_models/meta_classifier.h5"

# =========================
# STEP 1: LOAD BASE MODELS
# =========================

def load_base_models():
    models = []

    for cfg in BASE_MODELS:
        print(f"[INFO] Loading {cfg['name']} from {cfg['path']}")
        clf = cfg["class"](mode=cfg["mode"])
        clf.load(cfg["path"])
        model = clf.model

        # Safety (not strictly required, but good practice)
        model.trainable = False

        models.append({
            "name": cfg["name"],
            "model": model,
            "preprocess": get_preprocessing_function(cfg["class"])
        })

    return models

def build_meta_dataset(models):

    print("[INFO] Building meta-dataset from validation set")

    # Use preprocessing of FIRST model for generator
    datagen = ImageDataGenerator(
        preprocessing_function=models[0]["preprocess"]
    )

    generator = datagen.flow_from_directory(
        VAL_DATASET_DIR,
        target_size=INPUT_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['real', 'fake'],
        shuffle=False
    )

    y_true = generator.classes
    predictions = []

    for m in models:
        print(f"[INFO] Generating predictions for {m['name']}")
        preds = m["model"].predict(generator, verbose=1).flatten()
        predictions.append(preds)

    X_meta = np.stack(predictions, axis=1)
    y_meta = y_true

    print(f"[INFO] Meta features shape: {X_meta.shape}")
    return X_meta, y_meta

def train_meta_classifier(X, y):
    print("[INFO] Training meta-classifier")

    meta_model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(1, activation='sigmoid')
    ])

    meta_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    meta_model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    os.makedirs(os.path.dirname(META_MODEL_OUTPUT), exist_ok=True)
    meta_model.save(META_MODEL_OUTPUT)

    print(f"[INFO] Meta-classifier saved to {META_MODEL_OUTPUT}")

if __name__ == "__main__":
    base_models = load_base_models()
    X_meta, y_meta = build_meta_dataset(base_models)
    train_meta_classifier(X_meta, y_meta)
