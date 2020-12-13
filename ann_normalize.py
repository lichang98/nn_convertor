# -*- coding:utf-8 -*-
"""
Parameter normlization
Load trained model of ann, feed with sample data and 
apply parameter normalization on original ann network
"""
import tensorflow as tf
import keras
from keras import models
import numpy as np
import os
import fcn
from copy import deepcopy
import data_preprocessing


fcn_model_path = os.path.join(".", "ann_model_descs/fcn_model.h5")
conv_model_path = os.path.join(".", "ann_model_descs/conv_model.h5")
conv_bn_model_path = os.path.join(".", "ann_model_descs/conv_bn_model.h5")
fcn_norm_model_path = os.path.join(".", "ann_model_descs/fcn_normed_model.h5")
conv_norm_model_path = os.path.join(
    ".", "ann_model_descs/conv_normed_model.h5")
conv_bn_norm_model_path = os.path.join(
    ".", "ann_model_descs/conv_bn_normed_model.h5")

TYPE_FCN = 0
TYPE_CONV = 1
TYPE_CONV_BN = 2


def load_model(model_type: int) -> keras.models.Model:
    """
    Load already trainned model
    """
    if model_type == TYPE_FCN:
        return keras.models.load_model(fcn_model_path)
    elif model_type == TYPE_CONV:
        return keras.models.load_model(conv_model_path)
    else:
        return keras.models.load_model(conv_bn_model_path)


def param_normalization(model: keras.Model, dataX: np.array) -> keras.Model:
    previous_factor = 1
    beg_layer_idx = 0
    if model.layers[0].__class__.__name__ == "InputLayer":
        beg_layer_idx = 1

    layer_names = [layer.__class__.__name__ for layer in model.layers]
    print("layer names={}".format(layer_names))
    intermd_layers = []
    model.summary()
    if "BatchNormalization" in layer_names:
        prev_layer_idx_with_wt = beg_layer_idx
        for i in range(beg_layer_idx, len(model.layers)):
            if model.layers[i].__class__.__name__ == "BatchNormalization":
                print("prev layer index={}, weight shape={}".format(
                    prev_layer_idx_with_wt, np.shape(model.layers[prev_layer_idx_with_wt].get_weights())))
                print("bn var={}, sigma={}".format(np.shape(keras.backend.get_value(model.layers[i].moving_variance)),
                                                   np.shape(keras.backend.get_value(model.layers[i].gamma))))
                # applied bn operation to prev conv layer
                gamma = np.array(keras.backend.get_value(
                    model.layers[i].moving_variance), dtype="float32")
                variance = np.array(keras.backend.get_value(
                    model.layers[i].moving_variance), dtype="float32")
                adjust_factor = gamma / variance
                wts = deepcopy(
                    np.array(model.layers[prev_layer_idx_with_wt].get_weights(), dtype="float32"))
                wts = adjust_factor * wts
                model.layers[prev_layer_idx_with_wt].set_weights(wts)

            if len(np.shape(model.layers[i].get_weights())) > 1:
                prev_layer_idx_with_wt = i

        # remove bn layer
        intermd_layers.append(model.input)
        for layer in model.layers:
            if layer.__class__.__name__ == "BatchNormalization":
                continue
            else:
                intermd_layers.append(layer(intermd_layers[-1]))

    model = keras.models.Model(inputs=intermd_layers[0], outputs=intermd_layers[-1])
    model.summary()
    print("Start parameter normalization...")
    for i in range(beg_layer_idx, len(model.layers)):
        print("Current processing layer index {}, layer name {}".format(
            i, model.layers[i].__class__.__name__))
        if len(np.shape(model.layers[i].get_weights())) <= 1:
            continue
        print("Appliy scale to layer index {}, name={}".format(
            i, model.layers[i].__class__.__name__))
        max_wt, max_act = 0, 0
        input_weights = deepcopy(np.array(model.layers[i].get_weights()))
        max_wt = max(max_wt, np.max(input_weights))
        act_out = keras.models.Model(inputs=model.input, outputs=model.layers[i].output).predict(
            dataX, batch_size=1, verbose=1)
        max_act = max(max_act, np.max(act_out))
        scale_factor = max(max_wt, max_act)
        applied_factor = scale_factor/previous_factor
        model.layers[i].set_weights(input_weights/applied_factor)
        previous_factor = scale_factor

    return model


def test_ann(model: keras.models.Model, dataX: np.array, dataY: np.array):
    print("Start testing parameter normalization performace...")
    preds = model.predict(dataX, batch_size=1, verbose=1)
    preds = np.argmax(preds, axis=-1)
    accu = np.sum(preds == np.argmax(dataY, axis=-1))/len(preds)
    print("After parameter normalization, accuracy is {:.2%}".format(accu))


def save_normed_model(model: keras.models.Model, model_type: int):
    if model_type == TYPE_FCN:
        model.save(fcn_norm_model_path)
    elif model_type == TYPE_CONV:
        model.save(conv_norm_model_path)
    else:
        model.save(conv_bn_norm_model_path)


if __name__ == "__main__":
    # full connected network test
    # -------------------------
    # model = load_model(TYPE_FCN)
    # dataX, dataY = data_preprocessing.load_mnist_dataset()
    # normed_model = param_normalization(model, dataX[-50:])
    # test_ann(normed_model, dataX[:100], dataY[:100])
    # save_normed_model(normed_model,TYPE_FCN)

    # conv model network test
    # -------------------------
    # model = load_model(TYPE_CONV)
    # dataX, dataY = data_preprocessing.load_mnist_dataset(
    #     data_need_flatten=False)
    # normed_model = param_normalization(model, dataX[-50:])
    # test_ann(normed_model, dataX[:100], dataY[:100])
    # save_normed_model(normed_model, TYPE_CONV)

    # conv bn model network test
    # --------------------------
    model = load_model(TYPE_CONV_BN)
    dataX, dataY = data_preprocessing.load_mnist_dataset(
        data_need_flatten=False)
    normed_model = param_normalization(model, dataX[-50:])
    test_ann(normed_model, dataX[:100], dataY[:100])
    # save_normed_model(normed_model, TYPE_CONV_BN)
