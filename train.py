import os
from classifiers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from datetime import datetime
import numpy as np

def train_model(
    dataset_root,
    model_class=F3NetClassifier,
    mode='FAD',
    input_shape=(256, 256),
    batch_size=16,
    epochs=50,
    learning_rate=0.0001,
    patience=5,
    model_name='F3NetFAD',
    dataset_name='DeepFakes'
):
    # dataset paths
    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'valid')

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{model_name}_{dataset_name}_{timestamp}"

    #  outputs
    save_dir = os.path.join('data/saved_models', dataset_name, run_name)
    log_dir = os.path.join('data/logs', dataset_name, run_name)
    model_path = os.path.join(save_dir, f"{run_name}.h5")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    preprocessing_function = get_preprocessing_function(model_class=model_class)
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        zoom_range=0.2,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        channel_shift_range=30,
        horizontal_flip=True,
    )
    val_data_generator = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
    )

    train_generator = data_generator.flow_from_directory(
        train_dir,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='binary',
        classes=['real', 'fake']
    )

    val_generator = val_data_generator.flow_from_directory(
        val_dir,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='binary',
        classes=['real', 'fake'],
        shuffle=False
    )

    # model
    classifier = model_class(mode=mode, learning_rate=learning_rate)
    model = classifier.model

    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, restore_best_weights=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    lr_reduce = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )

    # training
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, tensorboard_callback, lr_reduce]
    )

    print(f"Model saved to: {model_path}")
    print(f"TensorBoard logs: {log_dir}")


if __name__ == '__main__':
    datasets = [
        ('../data/ffpp/datasets/DeepFakes_dataset', 'DeepFakes'),
        ('../data/ffpp/datasets/Face2Face_dataset', 'Face2Face'),
        ('../data/ffpp/datasets/FaceSwap_dataset', 'FaceSwap'),
        ('../data/ffpp/datasets/NeuralTextures_dataset', 'NeuralTextures'),
        ('../data/ffpp/datasets_augmented/combined', 'combined'),
    ]

    for dataset_path, dataset_name in datasets:
        print(f"\nStarting training for dataset: {dataset_name}\n")

        train_model(
            dataset_root=dataset_path,
            model_class=Meso4,
            mode='original',
            model_name='Meso4',
            dataset_name=dataset_name,
            batch_size=16,
            epochs=100,
            patience=5,
            learning_rate=0.001,
        )

        train_model(
            dataset_root=dataset_path,
            model_class=XceptionNet,
            mode='original',
            model_name='XceptionNet',
            dataset_name=dataset_name,
            batch_size=16,
            epochs=8,
            patience=3,
            learning_rate=0.0001,
        )

        train_model(
            dataset_root=dataset_path,
            model_class=F3NetClassifier,
            mode='FAD',
            model_name='F3NetFAD',
            dataset_name=dataset_name,
            batch_size=16,
            epochs=100,
            patience=5,
            learning_rate=0.0002,
        )
