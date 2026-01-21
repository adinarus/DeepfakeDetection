import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from eval import append_evaluation_to_excel
from classifiers import Meso4, XceptionNet, F3NetClassifier, get_preprocessing_function
from tqdm import tqdm

def evaluate_hybrid(base_models_cfg, meta_model_path, dataset_dir, dataset_name,
                    input_shape=(256,256), batch_size=16):
    """
    Evaluates a hybrid meta-classifier by computing predictions from base models
    sequentially (to save memory) and then feeding them to the meta-classifier.
    Includes progress bars for base models and hybrid predictions.
    """

    # --- Generator for dataset ---
    print(f"[INFO] Preparing dataset generator: {dataset_dir}")
    datagen = ImageDataGenerator()
    generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=input_shape,
        batch_size=1,
        class_mode='binary',
        classes=['real','fake'],
        shuffle=False
    )

    y_true = generator.classes
    num_samples = len(generator)
    num_base_models = len(base_models_cfg)

    # --- Compute predictions from each base model ---
    print(f"[INFO] Computing base model predictions for {num_samples} images...")
    base_preds = np.zeros((num_samples, num_base_models))

    for idx, cfg in enumerate(base_models_cfg):
        print(f"[INFO] Loading base model {cfg['class'].__name__}")
        clf = cfg["class"](mode=cfg["mode"])
        clf.load(cfg["path"])
        model = clf.model
        model.trainable = False
        preprocess = get_preprocessing_function(cfg["class"])

        for i in tqdm(range(num_samples), desc=f"{cfg['class'].__name__}", ncols=100):
            img, _ = generator[i]
            img_pre = preprocess(img.copy())
            pred = model.predict(img_pre, verbose=0)[0][0]
            base_preds[i, idx] = pred

        # free memory
        del model
        tf.keras.backend.clear_session()
        generator.reset()  # reset for next base model

    # --- Load meta-classifier ---
    print(f"[INFO] Loading meta-classifier: {meta_model_path}")
    meta_model = tf.keras.models.load_model(meta_model_path)

    # --- Predict with meta-classifier ---
    print("[INFO] Running meta-classifier predictions...")
    y_pred_meta = np.zeros(num_samples)
    for i in tqdm(range(num_samples), desc="Hybrid", ncols=100):
        img_vector = base_preds[i].reshape(1, -1)
        y_pred_meta[i] = meta_model.predict(img_vector, verbose=0)[0][0]

    y_pred_labels = (y_pred_meta > 0.5).astype(int)

    accuracy = np.mean(y_pred_labels == y_true)
    auc = roc_auc_score(y_true, y_pred_meta)
    report = classification_report(y_true, y_pred_labels, target_names=['real','fake'])
    conf_mat = confusion_matrix(y_true, y_pred_labels, labels=[1,0])

    print(f"\n[HYBRID] Dataset: {dataset_name}")
    print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    print(report)

    plt.figure()
    plt.imshow(conf_mat, cmap='Blues')
    plt.title('Confusion Matrix (Hybrid)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0,1], ['fake','real'])
    plt.yticks([0,1], ['fake','real'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, conf_mat[i,j], ha='center', va='center')
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_meta)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.title("ROC Curve (Hybrid)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    return {
        "accuracy": accuracy,
        "auc": auc,
        "y_true": y_true,
        "y_pred_labels": y_pred_labels,
        "y_pred_meta": y_pred_meta
    }

def log_hybrid_to_excel(results, meta_model_path, dataset_name, run_name="Hybrid_Run"):
    append_evaluation_to_excel(
        run_name=run_name,
        model_name="HybridMeta",
        mode="hybrid",
        dataset_name=dataset_name,
        model_path=meta_model_path,
        accuracy=results["accuracy"],
        auc=results["auc"],
        y_true=results["y_true"],
        y_pred=results["y_pred_labels"]
    )

if __name__ == '__main__':
    BASE_MODELS = [
        {"class": Meso4, "mode": "original", "path": r"data\saved_models\Meso4.h5"},
        {"class": XceptionNet, "mode": "original", "path": r"data\saved_models\Xception.h5"},
        {"class": F3NetClassifier, "mode": "FAD", "path": r"data\saved_models\F3NetFAD.h5"}
    ]

    META_MODEL = r"data\saved_models\meta_classifier.h5"

    results = evaluate_hybrid(
        base_models_cfg=BASE_MODELS,
        meta_model_path=META_MODEL,
        dataset_dir=r"C:\Users\Adi\Desktop\Licenta\datasets\ffpp\datasets_augmented\combined\10_percent_test_condition\noisy",
        dataset_name="noisy",
        input_shape=(256, 256),
        batch_size=16
    )
    log_hybrid_to_excel(results, META_MODEL, "noisy", run_name="Hybrid_sigmoid")
