import os
import random
import json
from PIL import Image
from tqdm import tqdm
import shutil
import numpy as np
from skimage import color


def collect_image_paths(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))
    return image_paths


def process_and_save_images(image_paths, output_dir, resolution, desc="Processing"):
    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(image_paths, desc=desc, unit="img"):
        file_name = os.path.basename(img_path)
        dest_path = os.path.join(output_dir, file_name)

        if resolution == 256:
            shutil.copy2(img_path, dest_path)
        else:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((resolution, resolution))
            img.save(dest_path)


def compute_lab_weights(image_paths, resolution, num_bins=64):
    print(f"Computing weights over LAB color space distribution for implementing weighted L1 loss")
    print(f"Processing all {len(image_paths)} training images in chunks...")

    chunk_size = 5000
    num_chunks = (len(image_paths) + chunk_size - 1) // chunk_size

    hist_accumulated = np.zeros((num_bins, num_bins), dtype=np.float64)
    a_edges = np.linspace(-128, 127, num_bins + 1)
    b_edges = np.linspace(-128, 127, num_bins + 1)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(image_paths))
        chunk_paths = image_paths[start_idx:end_idx]

        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (images {start_idx + 1}-{end_idx})")

        a_values = []
        b_values = []

        for img_path in tqdm(chunk_paths, desc=f"Chunk {chunk_idx + 1}/{num_chunks}"):
            img = Image.open(img_path).convert("RGB")
            if img.size != (resolution, resolution):
                img = img.resize((resolution, resolution))

            img_np = np.array(img) / 255.0
            lab = color.rgb2lab(img_np)
            a_values.extend(lab[:, :, 1].flatten())
            b_values.extend(lab[:, :, 2].flatten())

        hist_chunk, _, _ = np.histogram2d(a_values, b_values, bins=num_bins, range=[[-128, 127], [-128, 127]])
        hist_accumulated += hist_chunk

    epsilon = 1e-6
    weights = 1.0 / (hist_accumulated + epsilon)
    weights = weights / weights.mean()

    print(f"\nCompleted LAB weight computation using all {len(image_paths)} images")

    return weights.astype(np.float32), a_edges, b_edges


def generate_subset():
    with open("hyperparameters.json", "r") as f:
        config = json.load(f)

    train_size = config.get("train_size", 25000)
    val_size = config.get("val_size", 1000)
    test_size = config.get("test_size", 5000)
    resolution = config.get("resolution", 256)
    num_bins = config.get("lab_num_bins", 64)

    base_dir = "./places365"
    output_dir = "./data_subset"
    splits_data = {}

    for split, size in [("train", train_size), ("val", val_size), ("test", test_size)]:
        print(f"\nProcessing {split} split...")

        split_dir = os.path.join(base_dir, split)
        all_paths = collect_image_paths(split_dir)
        all_paths.sort()

        random.seed(42)
        selected_indices = random.sample(range(len(all_paths)), size)
        selected_paths = [all_paths[i] for i in selected_indices]

        splits_data[split] = selected_paths

    for split, paths in splits_data.items():
        output_split_dir = os.path.join(output_dir, split)
        process_and_save_images(paths, output_split_dir, resolution, desc=f"Saving {split} images")

    train_paths = splits_data["train"]
    weights, a_edges, b_edges = compute_lab_weights(train_paths, resolution, num_bins)

    print("\nSaving LAB weights...")
    np.savez("lab_weights.npz", weights=weights, a_edges=a_edges, b_edges=b_edges)


if __name__ == "__main__":
    generate_subset()