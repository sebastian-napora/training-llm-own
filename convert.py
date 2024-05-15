import tensorflow as tf
import json
import cv2

def preprocess_image(image_path):
    """Reads, converts, and preprocesses an image from the given path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array.
    """

    # Read image using OpenCV
    image = cv2.imread(image_path)

    # Handle potential errors during image reading
    if image is None:
        raise ValueError(f"Error reading image: {image_path}")

    # Convert to RGB and normalize pixel values (optional)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(float) / 255.0

    return image

with open('train-images.json', "r") as json_data:
    data = json.load(json_data)

images = []
labels = []

for image_info in data:
    label = image_info["label"]
    image_path = image_info["img"]

    try:
        # Preprocess image using the defined function
        image = preprocess_image(image_path)
        images.append(image.tolist())  # Convert to list for printing
        labels.append(label)
    except ValueError as e:
        print(f"Error processing image '{image_path}': {e}")

print("Images:")
print(images)
print("\nLabels:")
print(labels)
