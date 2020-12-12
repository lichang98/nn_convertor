# -*- coding:utf-8 -*-
"""
Parameter normlization
Load trained model of ann, feed with sample data and 
apply parameter normalization on original ann network
"""
import keras
import numpy as np
import os
import fcn
from copy import deepcopy
import data_preprocessing


fcn_model_path = os.path.join(".", "ann_model_descs/fcn_model.h5")
conv_model_path = os.path.join(".", "ann_model_descs/conv_model.h5")
fcn_norm_model_path = os.path.join(".", "ann_model_descs/fcn_normed_model.h5")
conv_norm_model_path = os.path.join(
    ".", "ann_model_descs/conv_normed_model.h5")

TYPE_FCN = 0
TYPE_CONV = 1


def load_model(model_type: int) -> keras.models.Model:
    """
    Load already trainned model
    """
    if model_type == TYPE_FCN:
        return keras.models.load_model(fcn_model_path)
    else:
        return keras.models.load_model(conv_model_path)


def param_normalization(model: keras.Model, dataX: np.array) -> keras.Model:
    previous_factor = 1
    beg_layer_idx = 0
    if model.layers[0].__class__.__name__ == "InputLayer":
        beg_layer_idx = 1
    print("Start parameter normalization...")
    for i in range(beg_layer_idx, len(model.layers)):
        print("Current processing layer index {}, layer name {}".format(
            i, model.layers[i].__class__.__name__))
        if len(model.layers[i].get_weights()) <= 1:
            continue
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


def save_normed_model(model: keras.models.Model):
    model.save(fcn_norm_model_path)


if __name__ == "__main__":
    # full connected network test
    # -------------------------
    # model = load_model()
    # dataX, dataY = data_preprocessing.load_mnist_dataset()
    # normed_model = param_normalization(model, dataX[-50:])
    # test_ann(normed_model, dataX[:100], dataY[:100])
    # save_normed_model(normed_model)

    # conv model network test
    # -------------------------
    model = load_model(TYPE_CONV)
    dataX, dataY = data_preprocessing.load_mnist_dataset(
        data_need_flatten=False)
    normed_model = param_normalization(model, dataX[-50:])
    test_ann(normed_model, dataX[:100], dataY[:100])
