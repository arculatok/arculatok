import numpy as np
import csv
import pandas as pd
import os
import tensorflow as tf
import random

words = ["bed", "yes"]

train_path = ".\\test_processed"
test_path = ".\\train_processed"
validation_path = ".\\val_processed"

EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 5
LEARNING_RATE = 0.0001

def load_data(dir):
    data = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            label = path.rstrip().split("\\")[2]
            if label in words:
                csv_file = pd.read_csv(path, sep=";")
                data.append((label, csv_file))
    random.shuffle(data)
    return data


train_data = load_data(train_path)
test_data = load_data(test_path)
validation_data = load_data(validation_path)


def build_model(input_shape, loss="sparse_categorical_crossentropy", learning_rate=0.0001):
    """Build neural network using keras.
    :param input_shape (tuple): Shape of array representing a sample train. E.g.: (44, 13, 1)
    :param loss (str): Loss function to use
    :param learning_rate (float):
    :return model: TensorFlow model
    """

    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3)

    # softmax output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])
    model.summary()

    return model

def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback])
    return history

train_labels = [x[0] for x in train_data]
train_mfcc = [x[1].to_numpy() for x in train_data]
test_labels = [x[0] for x in test_data]
test_mfcc = [x[1].to_numpy() for x in test_data]
validation_labels = [x[0] for x in validation_data]
validation_mfcc = [x[1].to_numpy() for x in validation_data]

print(train_labels)

input_shape = (train_mfcc[0].shape[0], train_mfcc[0].shape[1], 1)
model = build_model(input_shape)

history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, train_mfcc[0], train_labels, validation_mfcc[0], validation_labels)