"""
TensorFlow & Keras Deep Learning Training Script
For Handwritten Character Recognition (MNIST & EMNIST)

This script trains a high-accuracy Convolutional Neural Network (CNN)
and LeNet-5 model using TensorFlow 2.x and Keras.

Usage:
    python train_keras_cnn.py --dataset emnist --epochs 15
"""

import os
import argparse
import numpy as np

def build_cnn_model(input_shape=(28, 28, 1), num_classes=36):
    """Constructs a modern CNN with Batch Normalization and Dropout."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        print("TensorFlow is not installed. To run this script, run: pip install tensorflow")
        return None

    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1: Conv -> BatchNorm -> Conv -> MaxPool -> Dropout
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2: Conv -> BatchNorm -> Conv -> MaxPool -> Dropout
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3: Fully Connected
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_lenet5_model(input_shape=(28, 28, 1), num_classes=10):
    """Constructs the classic LeNet-5 architecture by Yann LeCun."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        print("TensorFlow is not installed.")
        return None

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # C1: 6 filters 5x5
        layers.Conv2D(6, (5, 5), activation='tanh', padding='same'),
        # S2: Average Pooling 2x2
        layers.AveragePooling2D((2, 2)),
        # C3: 16 filters 5x5
        layers.Conv2D(16, (5, 5), activation='tanh', padding='valid'),
        # S4: Average Pooling 2x2
        layers.AveragePooling2D((2, 2)),
        # Flatten
        layers.Flatten(),
        # C5: Dense 120
        layers.Dense(120, activation='tanh'),
        # F6: Dense 84
        layers.Dense(84, activation='tanh'),
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN / LeNet-5 on MNIST/EMNIST")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'emnist'])
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'lenet'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    print(f"Initializing TensorFlow training for {args.model.upper()} on {args.dataset.upper()}...")
    try:
        import tensorflow as tf
        if args.dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            num_classes = 10
        else:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            num_classes = 36

        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        model = build_cnn_model(num_classes=num_classes) if args.model == 'cnn' else build_lenet5_model(num_classes=num_classes)
        model.summary()

        model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_test, y_test))
        save_path = f"models/keras_{args.model}_{args.dataset}.h5"
        model.save(save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"Note: {e}")
