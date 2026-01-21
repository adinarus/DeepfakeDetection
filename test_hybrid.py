import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from classifiers import *

INPUT_SHAPE = (256, 256)

BASE_MODELS = [
    {
        "name": "Meso4",
        "class": Meso4,
        "mode": "original",
        "path": r"data\saved_models\Meso4.h5"
    },
    {
        "name": "Xception",
        "class": XceptionNet,
        "mode": "original",
        "path": r"data\saved_models\Xception.h5"
    },
    {
        "name": "F3NetFAD",
        "class": F3NetClassifier,
        "mode": "FAD",
        "path": r"data\saved_models\F3NetFAD.h5"
    }
]

META_MODEL_PATH = r"data\saved_models\meta_classifier.h5"

def test_image_hybrid(image_path):
    print("\n==============================")
    print(" HYBRID DEEPFAKE PREDICTION")
    print("==============================\n")

    base_predictions = []

    for cfg in BASE_MODELS:
        print(f"[INFO] Loading {cfg['name']}")

        classifier = cfg["class"](mode=cfg["mode"])
        classifier.load(cfg["path"])
        model = classifier.model

        img = load_img(image_path, target_size=INPUT_SHAPE)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        preprocess = get_preprocessing_function(cfg["class"])
        img = preprocess(img)

        pred = model.predict(img, verbose=0)[0][0]
        base_predictions.append(pred)

        label = "Fake" if pred > 0.5 else "Real"
        confidence = pred if pred > 0.5 else 1 - pred

        print(f"{cfg['name']:10s} → {label:4s} ({confidence*100:.2f}%)  raw={pred:.4f}")

    print("\n[INFO] Loading meta-classifier")
    meta_model = tf.keras.models.load_model(META_MODEL_PATH)

    base_predictions = np.array(base_predictions).reshape(1, -1)
    final_pred = meta_model.predict(base_predictions, verbose=0)[0][0]

    final_label = "Fake" if final_pred > 0.5 else "Real"
    final_confidence = final_pred if final_pred > 0.5 else 1 - final_pred

    print("\n------------------------------")
    print(f" FINAL HYBRID DECISION")
    print("------------------------------")
    print(f"Prediction: {final_label}")
    print(f"Confidence: {final_confidence*100:.2f}%")
    print(f"Raw score:  {final_pred:.4f}")
    print("------------------------------\n")

    return final_label, final_confidence


if __name__ == "__main__":
    IMAGE_PATH = r"test/FaceSwap_176_190_frame27.jpg"
    test_image_hybrid(IMAGE_PATH)
