import os
import numpy as np
from datetime import datetime
from matplotlib.pyplot import figure
from classifiers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt
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
    conf_mat = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, predictions)

    plt.figure()
    plt.imshow(conf_mat, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['real', 'fake'])
    plt.yticks([0, 1], ['real', 'fake'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, conf_mat[i, j], ha='center', va='center')

    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_base = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join('data/evaluation', dataset_name, model_base)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(1)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.figure(2)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))

    plt.close(figure(1))
    plt.close(figure(2))

    append_evaluation_to_excel(
        model_name=model_class.__name__,
        mode = mode,
        dataset_name=dataset_name,
        model_path=model_path,
        accuracy=accuracy,
        auc=auc,
        y_true=y_true,
        y_pred=y_pred,
        timestamp=timestamp
    )

    with open(os.path.join(output_dir, f'summary_{timestamp}.txt'), 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_class.__name__}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("Evaluation complete. Results saved to:", output_dir)

def append_evaluation_to_excel(
    model_name,
    mode,
    dataset_name,
    model_path,
    accuracy,
    auc,
    y_true,
    y_pred,
    timestamp,
    output_excel_path='data/evaluation/model_comparison.xlsx',
    sheet_name='All Results'
):

    report = classification_report(y_true, y_pred, target_names=['real', 'fake'], output_dict=True)

    row = {
        'Timestamp': timestamp,
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

if __name__ == '__main__':
    test_cases = [

        (r'data\saved_models\NeuralTextures\F3NetMixBlock_NeuralTextures_20250605-144142\F3NetMixBlock_NeuralTextures_20250605-144142.h5',
         r'..\data\ffpp\datasets\NeuralTextures_dataset\test', 'NeuralTextures', F3NetClassifier, 'Mix'),

        (r'data\saved_models\DeepFakes\F3NetMixBlock_updated_DeepFakes_20250605-175203\F3NetMixBlock_updated_DeepFakes_20250605-175203.h5',
         r'..\data\ffpp\datasets\DeepFakes_dataset\test', 'DeepFakes', F3NetClassifier, 'Mix'),
    ]

    for model_path, dataset_path, dataset_name, model_cls, model_mode in test_cases:
        print(f"Evaluating model {os.path.basename(model_path)} on dataset {dataset_path}\n")
        evaluate_model(model_path, dataset_path, dataset_name, model_class=model_cls, mode=model_mode)