import os
from PIL import Image, ImageEnhance
import numpy as np
import random

def apply_brightness(input_folder, output_folder, factor=1.5):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Applying brightness to: {filename}")
            img = Image.open(file_path)
            enhancer = ImageEnhance.Brightness(img)
            bright_img = enhancer.enhance(factor)
            bright_img.save(os.path.join(output_folder, filename))

def apply_compression(input_folder, output_folder, quality=25):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(input_folder, filename))
            img.save(os.path.join(output_folder, filename), quality=quality, optimize=True)

def add_noise(img, noise_level=10):
    img_array = np.array(img)
    noise = np.random.normal(0, noise_level, img_array.shape).astype(np.int16)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_gaussian_noise(input_folder, output_folder, noise_level=10):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(input_folder, filename)).convert('RGB')
            noisy_img = add_noise(img, noise_level)
            noisy_img.save(os.path.join(output_folder, filename))


input_path = r"..\..\data\ffpp\datasets_augmented\combined\test"
output_path = r"..\..\data\ffpp\datasets_augmented\combined\test_diff_conditions"

apply_brightness(os.path.join(input_path, 'real'), os.path.join(output_path, 'brightness', 'real'))
apply_brightness(os.path.join(input_path, 'fake'), os.path.join(output_path, 'brightness', 'fake'))
apply_compression(os.path.join(input_path, 'real'), os.path.join(output_path, 'compressed', 'real'))
apply_compression(os.path.join(input_path, 'fake'), os.path.join(output_path, 'compressed', 'fake'))
apply_gaussian_noise(os.path.join(input_path, 'real'), os.path.join(output_path, 'noisy', 'real'))
apply_gaussian_noise(os.path.join(input_path, 'fake'), os.path.join(output_path, 'noisy', 'fake'))
