import os
import cv2
import numpy as np
from classifiers import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils.pipeline import FaceFinder

input_shape = (256, 256)
default_classifier = Meso4
default_mode = 'original'

def test_image(model_path, image_path, model_class=default_classifier, mode=default_mode):
    print(f"Loading model: {model_class.__name__} at {model_path}")
    model = model_class(mode=mode)
    model.load(model_path)

    img = load_img(image_path, target_size=input_shape)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocess = get_preprocessing_function(model_class=model_class)
    img_array = preprocess(img_array)

    pred = model.predict(img_array)[0][0]
    label = "Fake" if pred > 0.5 else "Real"
    confidence = pred if pred > 0.5 else 1 - pred

    print(f"Prediction: {label} ({confidence * 100:.2f}%) actual prediction {pred}")
    return label, confidence

def test_video(model_path, video_path, model_class=default_classifier, mode=default_mode):
    print(f"Loading model: {model_class.__name__} | {model_path}")
    model = model_class(mode=mode)
    model.load(model_path)

    print(f"Processing video with FaceFinder: {video_path}")
    face_finder = FaceFinder(video_path, load_first_face=False)
    face_finder.find_faces(skipstep=15)

    frame_predictions = []
    count = 0

    for i in sorted(face_finder.coordinates.keys()):
        face = face_finder.get_aligned_face(i)
        if face is not None:
            face_resized = cv2.resize(face, input_shape)
            img_array = face_resized.astype("float32")
            img_array = np.expand_dims(img_array, axis=0)
            preprocess = get_preprocessing_function(model_class=model_class)
            img_array = preprocess(img_array)

            pred = model.predict(img_array)[0][0]
            frame_predictions.append(pred)
            print(f"Frame {i:04d} - Score: {pred:.4f}")
            count += 1

    if not frame_predictions:
        print("No faces were extracted from the video.")
        return "Unknown", 0.0

    average_score = np.mean(frame_predictions)
    label = "Fake" if average_score > 0.5 else "Real"
    confidence = average_score if average_score > 0.5 else 1 - average_score

    print("\nResults:")
    print(f"All frame scores: {[f'{p:.4f}' for p in frame_predictions]}")
    print(f"Average Score: {average_score:.4f}")
    print(f"Final Prediction: {label} ({confidence * 100:.2f}%) from {count} cropped faces.")
    return label, confidence

# if __name__ == '__main__':
#     label, confidence = test_video(r"test\Meso4.h5",r"test\000_003_nt.mp4")
