import os
import cv2
import json
from pipeline import FaceFinder

base_path = '../../data/videos'
dataset_output_base = '../../data/ffpp/datasets'

splits_path = os.path.join(base_path, 'splits')
original_videos_base = os.path.join(base_path, 'original_sequences', 'youtube', 'c23', 'videos')
methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

frame_interval_real = 10
frame_interval_fake = 10
faces_per_video_train = 50
faces_per_video_valid_test = 50
video_extensions = ('.mp4', '.avi', '.mov')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_faces_from_video(video_path, output_folder, interval=30, max_faces=200):
    """
    Extracts aligned face crops from the official splits, and saves all the extracted frames into a folder for each video.
    Next the dataset should be flattened.
    """
    video = FaceFinder(video_path, load_first_face=False)
    video.find_faces(skipstep=interval)

    count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    ensure_dir(video_output_folder)

    for frame_idx in sorted(video.coordinates.keys()):
        if count >= max_faces:
            break
        face = video.get_aligned_face(frame_idx)
        if face is not None:
            face = cv2.resize(face, (256, 256))
            filename = f"frame{count}.jpg"
            output_path = os.path.join(video_output_folder, filename)
            cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            count += 1

    print(f"Saved {count} faces for {video_name}")

def load_split(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def process_split(split_name, split_pairs):
    real_output_folder = os.path.join(dataset_output_base, split_name, 'real')
    ensure_dir(real_output_folder)

    for method in methods:
        method_output_folder = os.path.join(dataset_output_base, split_name, 'fake', method)
        ensure_dir(method_output_folder)

    max_faces = faces_per_video_train if split_name == 'train' else faces_per_video_valid_test

    total_videos = len(split_pairs)
    for idx, (real_id, fake_id) in enumerate(split_pairs):
        real_filename = f"{real_id}.mp4"

        # Process real video
        real_path = os.path.join(original_videos_base, real_filename)
        if os.path.exists(real_path):
            print(f"{split_name} [REAL] Processing {idx+1}/{total_videos}: {real_filename}")
            extract_faces_from_video(real_path, real_output_folder, interval=frame_interval_real, max_faces=max_faces)
        else:
            print(f"[WARNING] Missing REAL video: {real_filename}")

        # Process fakes
        for method in methods:
            fake_filename = f"{real_id}_{fake_id}.mp4"
            fake_path = os.path.join(base_path, 'manipulated_sequences', method, 'c23', 'videos', fake_filename)
            fake_output_folder = os.path.join(dataset_output_base, split_name, 'fake', method)

            if os.path.exists(fake_path):
                print(f"[{split_name}] [FAKE - {method}] Processing {idx+1}/{total_videos}: {fake_filename}")
                extract_faces_from_video(fake_path, fake_output_folder, interval=frame_interval_fake, max_faces=max_faces)
            else:
                print(f"[WARNING] Missing FAKE video ({method}): {fake_filename}")

def main():
    splits = {
        'train': os.path.join(splits_path, 'train.json'),
        'valid': os.path.join(splits_path, 'val.json'),
        'test': os.path.join(splits_path, 'test.json'),
    }

    for split_name, split_path in splits.items():
        print(f"Loading split: {split_name}")
        split_pairs = load_split(split_path)
        process_split(split_name, split_pairs)

if __name__ == '__main__':
    main()
