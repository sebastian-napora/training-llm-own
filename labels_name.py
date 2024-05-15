import os
from PIL import Image
import numpy as np
import json

def create_dataset_with_labels(data_dir, target_size=(32, 32)):
    """
    Creates a custom dataset structure with labels (winter: 6, summer: 9).

    Args:
        data_dir (str): Path to the directory containing labeled images.
            Images should be categorized into subfolders representing classes
            (either "winter" or "summer").
        target_size (tuple, optional): Desired size (width, height) of the images.
            Defaults to (32, 32).

    Returns:
        tuple: A tuple containing three elements:
            - pixel_values: A 4D NumPy array representing pixel values (RGB) for each image.
            - labels: A 1D NumPy array representing labels (6 for winter, 9 for summer).
            - image_paths: A list of image file paths.
    """

    # Validate input directory
    if not os.path.isdir(data_dir):
        raise ValueError(f"Invalid data directory: {data_dir}")

    # Create empty arrays for pixel values and labels
    pixel_values = []
    labels = []
    image_paths = []

    # Process images within each class directory
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        # Label assignment based on class name
        label = 6 if class_name == "winter" else 9  # Assign 6 for winter, 9 for summer

        # Process each image
        for image_path in images:
            img = Image.open(image_path)
            img = img.resize(target_size)  # Resize (optional preprocessing can be added here)
            img_array = np.array(img)  # Convert image to NumPy array
            pixel_values.append(img_array)  # Append the image array
            labels.append(label)  # Append the label
            image_paths.append(image_path)  # Track the image path

    # Convert lists to NumPy arrays for efficiency
    pixel_values = np.array(pixel_values, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)

    return pixel_values, labels, image_paths

def create_label_array(image_paths, label_mapping):
    """
    Creates a 2D NumPy array with labels for each image.

    Args:
        image_paths (list): List of image file paths.
        label_mapping (dict): Dictionary mapping image identifiers (e.g., filenames) to labels.

    Returns:
        np.array: A 2D NumPy array with each row representing a label for an image.
    """
    label_array = []
    for image_path in image_paths:
        image_id = os.path.basename(image_path)  # Get the filename as the identifier
        mapped_label = label_mapping.get(image_id)  # Get label from mapping using filename
        if mapped_label is None:
            raise ValueError(f"Missing label mapping for image: {image_id}")
        label_num = 6 if mapped_label == "winter" else 9
        label_array.append([label_num])  # Create a list with single label value

    return np.array(label_array, dtype=np.uint8)  # Convert to NumPy array

with open('train-images.json', "r") as json_data:
    data = json.load(json_data)

label_mapping = {
    os.path.basename(image["img"]): image["label"] for image in data
}

# Example usage (assuming you have a label mapping dictionary `label_mapping`)
data_dir = "./images"
pixel_values, labels, image_paths = create_dataset_with_labels(data_dir)

label_array = create_label_array(image_paths, label_mapping)

# Use image_paths to map labels correctly

# print("Pixel Values:")
# print(pixel_values.shape)  # Example output: (number_of_images, 32, 32, 3)
# print(pixel_values, 'images')
# print("\nLabels (as 2D array):")
# print(label_array.shape)  # Example output: (number_of_images, 1)
# print(label_array)  # Example output: [[mapped_label] [mapped_label] ...] (showing mapped labels)
