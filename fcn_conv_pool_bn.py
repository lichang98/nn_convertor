# -*- coding:utf-8 -*-
import numpy as np
import keras
import os
import data_preprocessing

model_save_path = os.path.join(".", "ann_model_descs/conv_bn_model.h5")
input_shape = (28, 28, 1)


def build_model(input_shape: tuple):
    """
    Create full connected network
    """
    x = keras.Input(shape=input_shape, dtype="float32")
    conv1 = keras.layers.Conv2D(512, (2, 2), strides=(
        2, 2), use_bias=False, activation="relu")(x)
    pool2 = keras.layers.MaxPool2D(strides=(2, 2))(conv1)
    conv3 = keras.layers.Conv2D(256, (1, 1), strides=(
        2, 2), use_bias=False, activation="relu")(pool2)
    pool4 = keras.layers.MaxPool2D(strides=(1, 1))(conv3)
    bn5 = keras.layers.BatchNormalization()(pool4)
    flatten6 = keras.layers.Flatten()(bn5)
    fc7 = keras.layers.Dense(512, activation="relu", use_bias=False)(flatten6)
    fc8 = keras.layers.Dense(256, activation="relu", use_bias=False)(fc7)
    fc9 = keras.layers.Dense(32, activation="relu", use_bias=False)(fc8)
    fc10 = keras.layers.Dense(10, activation="softmax", use_bias=False)(fc9)

    model = keras.models.Model(inputs=x, outputs=fc10)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    return model


def model_train(dataX: np.array, dataY: np.ndarray, model: keras.Model) -> keras.Model:
    """
    trainning network with training set
    @param dataX: M*N array float32, M is the number of samples, N is the total number of pixels of one image
    @param dataY: M*L array int32, where M is the number of samples, L is the number of different classes,
                    the one hot encoded labels for each image
    """
    model.fit(dataX, dataY, batch_size=1, epochs=3, validation_split=0.2)
    return model


def save_model(model: keras.Model):
    """
    save ann model with hdf5 format
    """
    model.save(model_save_path)


if __name__ == "__main__":
    model = build_model(input_shape)
    dataX, dataY = data_preprocessing.load_mnist_dataset(
        data_need_flatten=False)
    model = model_train(dataX, dataY, model)
    save_model(model)
