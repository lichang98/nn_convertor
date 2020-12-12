# -*- coding:utf-8 -*-
import numpy as np
import keras
import os
import data_preprocessing

model_save_path = os.path.join(".", "ann_model_descs/fcn_model.h5")
input_shape = (784,)


def build_model(input_shape: tuple):
    """
    Create full connected network
    """
    x = keras.Input(shape=input_shape, dtype="float32")
    fc1 = keras.layers.Dense(1024, activation="relu", use_bias=False)(x)
    fc2 = keras.layers.Dense(512, activation="relu", use_bias=False)(fc1)
    fc3 = keras.layers.Dense(32, activation="relu", use_bias=False)(fc2)
    fc4 = keras.layers.Dense(10, activation="softmax", use_bias=False)(fc3)

    model = keras.models.Model(inputs=x, outputs=fc4)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss="categorical_crossentropy", 
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
    dataX, dataY = data_preprocessing.load_mnist_dataset()
    model = model_train(dataX, dataY, model)
    save_model(model)
