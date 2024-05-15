import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import labels_name

# Load dataset
pixel_values, labels, image_paths = labels_name.create_dataset_with_labels('./images')

with open('train-images.json', "r") as json_data:
    data = json.load(json_data)

label_mapping = {
    os.path.basename(image["img"]): image["label"] for image in data
}

label_array = labels_name.create_label_array(image_paths, label_mapping)

train_images = pixel_values
train_labels = label_array

# Function to get random images
def randomImages(pixel_values):
    num_images = pixel_values.shape[0]

    # Randomly select four distinct indices from the range of number_of_images
    random_indices = np.random.choice(num_images, size=4, replace=False)

    # Retrieve the images at the selected indices
    random_images = pixel_values[random_indices]
    random_labels = label_array[random_indices]
    return random_images, random_labels

random_images, random_labels = randomImages(pixel_values)

test_images = random_images
test_labels = random_labels

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Plot random images
length = random_images.shape[0]
plt.figure(figsize=(10, 10))
for i, img in enumerate(random_images):
    plt.subplot(1, length, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.title(f"Random Image {i+1}")
    plt.axis('off')
plt.show()

# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Single unit with sigmoid activation for binary classification

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Plot accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
