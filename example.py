import numpy as np
from classifiers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
#classifier = MesoInception4()
#classifier = Meso4()
classifier = F3NetClassifier(mode='Mix')
# classifier.load('weights/Meso4_DF.h5')
# classifier.load('saved_models/mesoinception_ffpp.h5')
classifier.load(r'saved_models\NeuralTextures\F3NetMixBlock_NeuralTextures_20250605-093152\F3NetMixBlock_NeuralTextures_20250605-093152.h5')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        r'D:\Licenta\DeepfakeDetection\data\ffpp\datasets\DeepFakes_dataset\test\fake\000_003_frame0.jpg',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        classes=['real', 'fake']
)

# 3 - Predict
# X, y = next(generator)
# print('Predicted :', classifier.predict(X), '\nReal class :', y)
for i in range(len(generator)):
    X, y = next(generator)
    pred = classifier.predict(X)
    print(f"Image {i+1}: Predicted = {pred[0][0]:.4f}, Real class = {int(y[0])}")

print("Image prediction complete. Now switching to video classification...")
# 4 - Prediction for a video dataset
#
# classifier.load('weights/Meso4_F2F.h5')
#
# predictions = compute_accuracy(classifier, 'test_videos')
# if predictions:
#     for video_name in predictions:
#         # print(f"`{video_name}` video class prediction :", predictions[video_name][0])
#         mean_score = predictions[video_name][0]
#         raw_scores = predictions[video_name][1]
#         print(f"`{video_name}` video class prediction: {mean_score:.4f}")
#         print("Per-frame scores:", raw_scores.reshape(-1))
# else:
#     print("No videos were found or processed.")
