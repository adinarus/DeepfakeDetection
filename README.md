# Deepfake Detection - MesoNet, XceptionNet, F3Net
This project implements a deepfake detection pipeline for images, using both classical (Meso4, MesoInception4) and modern deep learning models (F3Net, XceptionNet) using Tensorflow.

---

## Implemented Models and Sources

### MesoNet (Meso4, MesoInception4)
- Lightweight CNNs designed for deepfake detection
- Paper: [MesoNet: a Compact Facial Video Forgery Detection Network](https://arxiv.org/abs/1809.00888)
- [Official code](https://github.com/DariusAf/MesoNet)

### XceptionNet
- Deep CNN with depthwise separable convolutions. It is used as a baseline for face forgery detection.
- Paper: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

### F3Net (Frequency-aware Fusion Network)
- A novel model that combines Frequency-aware Decomposition (FAD) and Local Frequency Statistics (LFS), built on a custom Xception pre-trained on Imagenet.
- Paper: [Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues](https://arxiv.org/abs/2007.09355)
- Re-implemented in TensorFlow based on this [implementation in Pytorch](https://github.com/yyk-wew/F3Net)

## Dataset

The models have been trained on the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset using the official splits for fair benchmarking.

## Requirements

Python 3.10.16

Tensorflow 2.10.0

CUDA version 11.2

cuDNN version: 8.1

