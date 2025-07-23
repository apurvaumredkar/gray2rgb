import os
import random
import json
from PIL import Image
from tqdm import tqdm

def collect_image_paths(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, f))
    return image_paths

def copy_and_resize_images(rgb_paths, rgb_dest_dir, resolution, desc=None, use_tqdm=True):
    os.makedirs(rgb_dest_dir, exist_ok=True)
    iterator = rgb_paths
    if use_tqdm:
        iterator = tqdm(rgb_paths, total=len(rgb_paths), desc=desc, unit="img")
    for rgb_path in iterator:
        file_name = os.path.basename(rgb_path)
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_img = rgb_img.resize((resolution, resolution))
        rgb_img.save(os.path.join(rgb_dest_dir, file_name))

def process_split(base_dir, output_rgb_dir, split, n_images, resolution):
    split_dir = os.path.join(base_dir, split)
    all_paths = collect_image_paths(split_dir)
    all_paths.sort()
    total = len(all_paths)
    if n_images > total:
        raise ValueError(f"Requested {n_images} images but only {total} available in {split}.")
    random.seed(42)
    selected_indices = random.sample(range(total), n_images)
    selected_rgb_paths = [all_paths[i] for i in selected_indices]
    rgb_dest_dir = os.path.join(output_rgb_dir, split)
    print(f"[{split}] Processing {n_images} images...")
    copy_and_resize_images(selected_rgb_paths, rgb_dest_dir, resolution, desc=f"Processing {split} images")
    print(f"[{split}] Done.")

def generate_subset():
    with open("hyperparameters.json", "r") as f:
        hparams = json.load(f)
    split_sizes = {
        "train": hparams.get("train_size", 50000),
        "val": hparams.get("val_size", 10000),
        "test": hparams.get("test_size", 15000)
    }
    resolution = hparams.get("resolution", 256)
    base_dir = "./places365"
    output_rgb_dir = "./data_subset"
    splits = ["train", "val", "test"]
    for split in splits:
        process_split(base_dir, output_rgb_dir, split, split_sizes[split], resolution)
    print("\nAll splits processed.")

if __name__ == "__main__":
    generate_subset()
