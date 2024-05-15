import os
from PIL import Image
import random  # Replace with a more robust splitting method if needed

def create_dataset(data_dir, target_size=(32, 32), split_ratio=0.8):
  """
  Creates a custom dataset structure similar to CIFAR-10.

  Args:
      data_dir (str): Path to the directory containing labeled images.
          Images should be categorized into subfolders representing classes.
      target_size (tuple, optional): Desired size (width, height) of the images.
          Defaults to (32, 32).
      split_ratio (float, optional): Ratio for splitting data into training
          and testing sets. Defaults to 0.8 (80% training).
  """

  # Validate input directory
  if not os.path.isdir(data_dir):
    raise ValueError(f"Invalid data directory: {data_dir}")

  # Create directory structure for training and testing sets
  train_dir = os.path.join(data_dir, "train")
  test_dir = os.path.join(data_dir, "test")
  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(test_dir, exist_ok=True)

  # Process images within each class directory
  for class_name in os.listdir(data_dir):
    if not os.path.isdir(os.path.join(data_dir, class_name)):
      continue

    class_dir = os.path.join(data_dir, class_name)
    images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

    # Split images into training and testing sets (consider using scikit-learn later)
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Preprocess images (resize, normalize, etc.)
    for image_path in train_images:
      img = Image.open(image_path)
      img = img.resize(target_size, Image.Resampling.LANCZOS)  # Resize
      # Add additional preprocessing steps here (normalization, etc.)
      new_path = os.path.join(train_dir, class_name, os.path.basename(image_path))
      os.makedirs(os.path.dirname(new_path), exist_ok=True)  # Create subdirectories if needed
      img.save(new_path)

    for image_path in test_images:
      img = Image.open(image_path)
      img = img.resize(target_size,  Image.Resampling.LANCZOS)  # Resize
      # Add additional preprocessing steps here (normalization, etc.)
      new_path = os.path.join(test_dir, class_name, os.path.basename(image_path))
      os.makedirs(os.path.dirname(new_path), exist_ok=True)  # Create subdirectories if needed
      img.save(new_path)

  print(f"Successfully created custom dataset in: {data_dir}")

# Example usage (replace with your actual data directory)
data_dir = "./images"
create_dataset(data_dir)
