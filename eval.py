import os
import numpy as np
from datetime import datetime
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from classifiers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
import pandas as pd


def evaluate_model(model_path, dataset_dir, dataset_name, model_class=F3NetClassifier, mode='Mix', input_shape=(256, 256), batch_size=16):
    classifier = model_class(mode=mode)
    classifier.load(model_path)
    model = classifier.model

    preprocessing_function = get_preprocessing_function(model_class=model_class)
    datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='binary',
        classes=['real', 'fake'],
        shuffle=False
    )

    predictions = model.predict(generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = generator.classes

    accuracy = np.mean(y_pred == y_true)
    report = classification_report(y_true, y_pred, target_names=['real', 'fake'])
    auc = roc_auc_score(y_true, predictions)

    conf_mat = confusion_matrix(y_true, y_pred, labels=[1, 0])  # 1 = fake, 0 = real must be in order to have true positive in the upper left corner

    plt.figure()
    plt.imshow(conf_mat, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted value')
    plt.ylabel('Actual value')

    plt.xticks([0, 1], ['fake', 'real'])
    plt.yticks([0, 1], ['fake', 'real'])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, conf_mat[i, j], ha='center', va='center')

    fpr, tpr, thresholds = roc_curve(y_true, predictions)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    model_base = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join('data/evaluation', model_base)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(1)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_confusion_matrix.png'))
    plt.figure(2)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_roc_curve.png'))

    plt.close(figure(1))
    plt.close(figure(2))

    append_evaluation_to_excel(
        run_name=model_base,
        model_name=model_class.__name__,
        mode = mode,
        dataset_name=dataset_name,
        model_path=model_path,
        accuracy=accuracy,
        auc=auc,
        y_true=y_true,
        y_pred=y_pred
    )

    with open(os.path.join(output_dir, f'{dataset_name}_summary.txt'), 'w') as f:
        f.write(f"Model: {model_class.__name__}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("Evaluation complete. Results saved to:", output_dir)

def append_evaluation_to_excel(
    run_name,
    model_name,
    mode,
    dataset_name,
    model_path,
    accuracy,
    auc,
    y_true,
    y_pred,
    output_excel_path='data/evaluation/model_comparison.xlsx',
    sheet_name='All Results'
):

    report = classification_report(y_true, y_pred, target_names=['real', 'fake'], output_dict=True)

    row = {
        'Run name': run_name,
        'Model': model_name,
        'Mode' : mode,
        'Dataset': dataset_name,
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Prec (Real)': round(report['real']['precision'], 4),
        'Prec (Fake)': round(report['fake']['precision'], 4),
        'Recall (Real)': round(report['real']['recall'], 4),
        'Recall (Fake)': round(report['fake']['recall'], 4),
        'F1 (Real)': round(report['real']['f1-score'], 4),
        'F1 (Fake)': round(report['fake']['f1-score'], 4),
        'Avg Prec': round(report['weighted avg']['precision'], 4),
        'Avg Recall': round(report['weighted avg']['recall'], 4),
        'F1 Score': round(report['weighted avg']['f1-score'], 4),
        'File Path': model_path
    }

    if os.path.exists(output_excel_path):
        df = pd.read_excel(output_excel_path, sheet_name=sheet_name)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    with pd.ExcelWriter(output_excel_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

def compare_models_roc(model_list, dataset_dir, dataset_name, input_shape=(256, 256), batch_size=16):
    plt.figure(figsize=(8, 6))

    for model_name, model_path, model_class, mode in model_list:
        classifier = model_class(mode=mode)
        classifier.load(model_path)
        model = classifier.model

        preprocessing_function = get_preprocessing_function(model_class=model_class)
        datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        generator = datagen.flow_from_directory(
            dataset_dir,
            target_size=input_shape,
            batch_size=batch_size,
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=False
        )

        predictions = model.predict(generator)

        # roc curve and auc
        y_true = generator.classes
        fpr, tpr, thresholds = roc_curve(y_true, predictions)
        auc = roc_auc_score(y_true, predictions)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.4f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f'ROC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save results and graph
    output_dir = os.path.join('data/evaluation', 'roc_comparison')
    os.makedirs(output_dir, exist_ok=True)
    roc_path = os.path.join(output_dir, f'roc_{dataset_name}.png')
    plt.savefig(roc_path)
    plt.close()


if __name__ == '__main__':

    dataset_test = [
        [r'..\data\ffpp\datasets\DeepFakes_dataset\test', r'DeepFakes'],
        [r'..\data\ffpp\datasets\Face2Face_dataset\test', r'Face2Face'],
        [r'..\data\ffpp\datasets\FaceSwap_dataset\test', r'FaceSwap'],
        [r'..\data\ffpp\datasets\NeuralTextures_dataset\test', r'NeuralTextures'],
        [r'..\data\ffpp\datasets_augmented\combined\test', 'combined']
        # [r'..\data\ffpp\datasets_augmented\combined\test_diff_conditions\compressed', r'combined_compressed'],
        # [r'..\data\ffpp\datasets_augmented\combined\test_diff_conditions\brightness', r'combined_brightness'],
        # [r'..\data\ffpp\datasets_augmented\combined\test_diff_conditions\noisy', r'combined_noisy']
    ]

    test_cases = []

    for dataset_path, dataset_name in dataset_test:

        test_cases.append(
            (r'data\saved_models\combined\Meso4_combined_20250601-122251\Meso4_combined_20250601-122251.h5',
            dataset_path,
            dataset_name,
            Meso4,
            'original')
        )

    for model_path, dataset_path, dataset_name, model_cls, model_mode in test_cases:
        print(f"Evaluating model {os.path.basename(model_path)} on dataset {dataset_path}\n")
        evaluate_model(model_path, dataset_path, dataset_name, model_class=model_cls, mode=model_mode)
