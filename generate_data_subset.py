import os
import random
import shutil
import argparse
from PIL import Image


def count_images(folder_path):
    count = 0
    for root, _, files in os.walk(folder_path):
        count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return count


def collect_image_paths(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, f))
    return image_paths


def copy_and_resize_images(image_paths, dest_dir, resolution, convert_to_gray=False):
    os.makedirs(dest_dir, exist_ok=True)
    for src_path in image_paths:
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(dest_dir, file_name)

        img = Image.open(src_path).convert("RGB")
        img = img.resize((resolution, resolution))

        if convert_to_gray:
            img = img.convert("L")

        img.save(dst_path)


def generate_subset(base_dir, output_rgb_dir, output_gray_dir, split_sizes, resolution):
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    print("Counting images...")
    print(f"Train: {count_images(train_dir)}, Val: {count_images(val_dir)}, Test: {count_images(test_dir)}")

    print("Collecting image paths...")
    train_images = collect_image_paths(train_dir)
    val_images = collect_image_paths(val_dir)
    test_images = collect_image_paths(test_dir)

    random.seed(42)
    selected_train = random.sample(train_images, split_sizes["train"])
    selected_val = random.sample(val_images, split_sizes["val"])
    selected_test = random.sample(test_images, split_sizes["test"])

    # RGB subset creation
    print("Creating RGB subsets...")
    for split_name, images in zip(["train", "val", "test"], [selected_train, selected_val, selected_test]):
        dest_dir = os.path.join(output_rgb_dir, split_name)
        copy_and_resize_images(images, dest_dir, resolution, convert_to_gray=False)
        print(f"Copied {len(images)} RGB images to {dest_dir}")

    # Grayscale subset creation
    print("Creating grayscale subsets...")
    for split_name, images in zip(["train", "val", "test"], [selected_train, selected_val, selected_test]):
        dest_dir = os.path.join(output_gray_dir, split_name)
        copy_and_resize_images(images, dest_dir, resolution, convert_to_gray=True)
        print(f"Copied {len(images)} grayscale images to {dest_dir}")

    print("\nSubset generation complete.")
    print(f"Final RGB image counts — Train: {count_images(os.path.join(output_rgb_dir, 'train'))}, "
          f"Val: {count_images(os.path.join(output_rgb_dir, 'val'))}, "
          f"Test: {count_images(os.path.join(output_rgb_dir, 'test'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RGB and grayscale subsets from Places365.")

    parser.add_argument("--base_dir", type=str, default="./places365",
                        help="Root directory of original Places365 dataset.")
    parser.add_argument("--output_rgb_dir", type=str, default="./data_subset/rgb",
                        help="Directory to save RGB subset.")
    parser.add_argument("--output_gray_dir", type=str, default="./data_subset/gray",
                        help="Directory to save grayscale subset.")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Output image resolution (square).")
    parser.add_argument("--train_size", type=int, default=10000,
                        help="Number of training images.")
    parser.add_argument("--val_size", type=int, default=1000,
                        help="Number of validation images.")
    parser.add_argument("--test_size", type=int, default=2500,
                        help="Number of test images.")

    args = parser.parse_args()

    split_sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size
    }

    generate_subset(args.base_dir, args.output_rgb_dir, args.output_gray_dir, split_sizes, args.resolution)
