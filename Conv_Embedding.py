import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

################# Conv 4 Feature Extractor #############################################

def Conv4FeatureExtractor(input_shape=(32, 32, 3)):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), padding='same', activation=None, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation=None),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation=None),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation=None),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten()
    ])
    return model

# Example usage
if __name__ == "__main__":
    model = Conv4FeatureExtractor()
    model.summary()


################# ResNet12 Feature Extractor #############################################

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):
    """A residual block for ResNet-12."""
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = x

    # First convolution
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolution
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Add shortcut and apply ReLU
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

def build_resnet12(input_shape):
    """Build the ResNet-12 feature extractor."""
    inputs = tf.keras.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    x = residual_block(x, filters=64, stride=1)
    x = residual_block(x, filters=64, stride=1)
    x = residual_block(x, filters=64, stride=1)

    x = residual_block(x, filters=128, stride=2)
    x = residual_block(x, filters=128, stride=1)
    x = residual_block(x, filters=128, stride=1)

    x = residual_block(x, filters=256, stride=2)
    x = residual_block(x, filters=256, stride=1)
    x = residual_block(x, filters=256, stride=1)

    x = residual_block(x, filters=512, stride=2)
    x = residual_block(x, filters=512, stride=1)
    x = residual_block(x, filters=512, stride=1)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Output embedding
    model = models.Model(inputs, x)
    return model

input_shape = (32, 32, 3)  # Input shape for CIFAR-FS
resnet12 = build_resnet12(input_shape)
resnet12.summary()
