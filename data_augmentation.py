# Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import pickle
import os

# Create folder for datasets, model files, results
project_dir = r"C:\Users\jackw\Documents\MAAE4904\Project"

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = len(np.unique(y_train))  # Determine the number of classes (10 for CIFAR-10)

# Define augmentation functions
def horizontal_flip(image):
    """Applies horizontal flip to the input image."""
    return tf.image.flip_left_right(image)

def vertical_flip(image):
    """Applies vertical flip to the input image."""
    return tf.image.flip_up_down(image)

def random_rotation(image):
    """Applies random rotation to the input image."""
    return tf.image.rot90(image, k=np.random.choice([1, 2, 3]))  # Rotate 90, 180, or 270 degrees

def adjust_brightness(image):
    """Adjusts the brightness of the input image randomly."""
    return tf.image.adjust_brightness(image, delta=np.random.uniform(-0.3, 0.3))

# List of augmentation functions
augmentation_functions = [horizontal_flip, vertical_flip, random_rotation, adjust_brightness]

# Initialize lists to store augmented data
x_augmented = []
y_augmented = []

# Perform data augmentation for each class
for class_label in range(num_classes):
    class_indices = np.where(y_train == class_label)[0]
    class_images = x_train[class_indices]
    class_labels = y_train[class_indices]

    required_augmentations = 5500 - class_images.shape[0]  # Target of 5500 images per class

    for _ in range(required_augmentations):
        random_index = np.random.randint(0, class_images.shape[0])
        original_image = class_images[random_index]
        label = class_labels[random_index]

        # Randomly select and apply an augmentation function
        augmentation_function = np.random.choice(augmentation_functions)
        augmented_image = augmentation_function(original_image).numpy()

        # Append augmented data
        x_augmented.append(augmented_image)
        y_augmented.append(label)

# Combine original and augmented data
x_train_augmented = np.concatenate((x_train, np.array(x_augmented)), axis=0)
y_train_augmented = np.concatenate((y_train, np.array(y_augmented)), axis=0)

# Save the augmented dataset
with open(os.path.join(project_dir,"train_augmented.pkl"), "wb") as f:
    pickle.dump({"x_train_augmented": x_train_augmented, "y_train_augmented": y_train_augmented}, f)

print(f"Final augmented dataset shape: {x_train_augmented.shape}")
print(f"Final augmented labels shape: {y_train_augmented.shape}")
print("Augmented dataset saved as 'train_augmented.pkl'")

# Visualize original and augmented images
def display_original_and_augmented(class_label, num_pairs=5):
    """Displays pairs of original and augmented images for a specific class."""
    class_indices = np.where(y_train == class_label)[0]

    fig, axes = plt.subplots(num_pairs, 2, figsize=(8, num_pairs * 4))
    fig.suptitle(f"Original and Augmented Images for Class {class_label}", fontsize=16)

    for i in range(num_pairs):
        random_index = np.random.choice(class_indices)
        original_image = x_train[random_index]
        augmentation_function = np.random.choice(augmentation_functions)
        augmented_image = augmentation_function(original_image).numpy()

        # Display original and augmented images
        axes[i, 0].imshow(original_image.astype("uint8"))
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(augmented_image.astype("uint8"))
        axes[i, 1].set_title("Augmented Image")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

# Display examples for a few classes
for class_label in range(3):  # Example: Display for first 3 classes
    display_original_and_augmented(class_label, num_pairs=3)