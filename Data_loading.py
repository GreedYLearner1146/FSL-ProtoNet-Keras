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

# Split into train, val, and test classes, in accordance to the FC-100 config in TADAM. The number represents the class indices.
train_classes_idx = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                     31,32,33,34,46,47,48,49,50,51,52,53,54,55,76,77,78,79,80,86,87,88,89,90,
                     91,92,93,94,95,96,97,98,99,100]
val_classes_idx = [41,42,43,44,45,56,57,58,59,60,66,67,68,69,70,81,82,83,84,85]
test_classes_idx = [0,1,2,3,4,36,37,38,39,40,61,62,63,64,65,71,72,73,74,75]

# Filter data for each split
def filter_data(x, y, classes):
        mask = np.isin(y, classes)  # True where an element of y is in classes also and False otherwise.
        return x[mask], y[mask]

x_train_fs, y_train_fs = filter_data(x_train, y_train, train_classes_idx)  # Train img and label.
x_val_fs, y_val_fs = filter_data(x_train, y_train, val_classes_idx)       # Valid img and label.
x_test_fs, y_test_fs = filter_data(x_train, y_train, test_classes_idx)    # Test img and label.

