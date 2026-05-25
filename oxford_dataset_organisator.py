import os
import random
import shutil
from pathlib import Path


def organize_and_split_cats(raw_images_dir, output_dir, train_ratio=0.8, copy_files=True):
    """
    Filters out dogs, groups cat images by breed, and splits them into
    train and test directories with an 80/20 ratio.

    """
    raw_path = Path(raw_images_dir)
    out_path = Path(output_dir)

    random.seed(42)

    breed_images = {}

    for img_file in raw_path.glob("*.jpg"):
        filename = img_file.name
        if "_" not in filename:
            continue

        breed_name, _ = filename.rsplit("_", 1)

        # Dataset rule: First letter capitalized = Cat breed
        if breed_name[0].isupper():
            if breed_name not in breed_images:
                breed_images[breed_name] = []
            breed_images[breed_name].append(img_file)

    total_train_images = 0
    total_test_images = 0

    for breed, images in breed_images.items():
        # Shuffle the list of images for this specific breed
        random.shuffle(images)

        # Calculate split index
        split_idx = int(len(images) * train_ratio)
        train_pool = images[:split_idx]
        test_pool = images[split_idx:]

        train_target_dir = out_path / "train" / breed
        test_target_dir = out_path / "test" / breed

        train_target_dir.mkdir(parents=True, exist_ok=True)
        test_target_dir.mkdir(parents=True, exist_ok=True)

        def distribute_files(file_list, destination_dir):
            count = 0
            for img_path in file_list:
                destination = destination_dir / img_path.name
                if copy_files:
                    shutil.copy(img_path, destination)
                else:
                    shutil.move(img_path, destination)
                count += 1
            return count

        total_train_images += distribute_files(train_pool, train_target_dir)
        total_test_images += distribute_files(test_pool, test_target_dir)

    print(" Processing complete!")
    print(f" Cat Breeds Found: {len(breed_images)}")
    print(f"️  Training images copies: {total_train_images}")
    print(f" Testing images copies: {total_test_images}")
    print(f" Structured dataset saved to: {out_path.resolve()}")


RAW_IMAGES_DIR = "./images/images"
STRUCTURED_OUTPUT_DIR = "./images/structured"

if __name__ == "__main__":
    organize_and_split_cats(RAW_IMAGES_DIR, STRUCTURED_OUTPUT_DIR, train_ratio=0.8, copy_files=True)