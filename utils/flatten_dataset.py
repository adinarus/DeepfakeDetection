import os
import shutil

def merge_subfolders_into_one(input_folder, output_folder):
    """
    Copies images from subfolders into a single output folder, renaming them in the process to preserve their origin.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    method_type = os.path.basename(os.path.normpath(input_folder))

    for folder in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            new_name = f"{method_type}_{folder}_{file}"
            src = os.path.join(folder_path, file)
            dst = os.path.join(output_folder, new_name)

            shutil.copy2(src, dst)
            print(f"{new_name}")

    print(f"\nimages saved to: {output_folder}")

if __name__ == '__main__':
    input_folder = r"../../data/ffpp/datasets/DeepFakes/valid/real"
    output_folder = r"../../data/ffpp/datasets/DeepFakes_merged/valid/real"
    merge_subfolders_into_one(input_folder, output_folder)
