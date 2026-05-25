import os
import shutil
from pathlib import Path


def organize_oxford_dataset(raw_images_dir, output_dir, copy_files=True):
    """
    Organizes the Oxford-IIIT Pet dataset into structured folders
    """
    raw_path = Path(raw_images_dir)
    out_path = Path(output_dir)

    # Counter for tracking progress
    moved_count = 0

    for img_file in raw_path.glob("*.jpg"):
        filename = img_file.name
        if "_" not in filename:
            continue
        breed_name, _ = filename.rsplit("_", 1)

        # Oxford dataset convention:
        # First letter capitalized = Cat breed
        # First letter lowercase = Dog breed
        if breed_name[0].isupper():
            species = "cats"
        else:
            species = "dogs"

        # Define the new target directory path
        target_dir = out_path / species / breed_name
        target_dir.mkdir(parents=True, exist_ok=True)

        destination = target_dir / filename

        # Move or copy the file
        if copy_files:
            shutil.copy(img_file, destination)
        else:
            shutil.move(img_file, destination)

        moved_count += 1

    print(f" Processing complete! Total images organized: {moved_count}")
    print(f" Structured dataset saved to: {out_path.resolve()}")


RAW_IMAGES_DIR = "./images/images"
STRUCTURED_OUTPUT_DIR = "./images/structured"

if __name__ == "__main__":
    organize_oxford_dataset(RAW_IMAGES_DIR, STRUCTURED_OUTPUT_DIR, copy_files=True)