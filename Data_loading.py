import tensorflow as tf
import numpy as np

# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# CIFAR-100 has 100 classes, each with 500 training and 100 test images
print("Training data shape:", x_train.shape)  # (50000, 32, 32, 3)
print("Training labels shape:", y_train.shape)  # (50000, 1)
print("Test data shape:", x_test.shape)  # (10000, 32, 32, 3)
print("Test labels shape:", y_test.shape)  # (10000, 1)

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Split into train, val, and test classes. The number represents the class indices.
train_classes_idx = [30, 46, 66, 97, 0, 43, 32,  4, 99, 16, 86 , 57,  50, 39, 18, 36, 14, 67, 81,  8, 96, 93, 75, 53, 6, 22, 15, 55, 92,  3, 79,
                     51, 21, 34,  2, 41, 13, 62, 49,  5, 58,  7,  9, 98, 45, 56 ,20, 63, 28, 83, 29, 17, 60, 69, 90, 87, 88, 64, 74, 47]

val_classes_idx = [37, 23, 38, 33, 89, 72, 77, 52, 65, 10, 12, 78, 85, 25, 31, 19, 59, 40, 42, 35]

test_classes_idx = [71, 82, 44, 84, 68, 94, 91, 70, 73, 61, 24, 27, 26, 80, 95, 54,  1, 76, 11, 48]

# Filter data for each split
def filter_data(x, y, classes):
        mask = np.isin(y, classes)  # True where an element of y is in classes also and False otherwise.
        return x[mask], y[mask]

x_train_fs, y_train_fs = filter_data(x_train, y_train, train_classes_idx)  # Train img and label.
x_val_fs, y_val_fs = filter_data(x_train, y_train, val_classes_idx)       # Valid img and label.
x_test_fs, y_test_fs = filter_data(x_train, y_train, test_classes_idx)    # Test img and label.

