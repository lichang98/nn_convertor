# -*- coding:utf-8 -*-
"""
Convert the float point parameters of ANN model to fix point bitwidth
After weight normalization and tested, this file provide method for 
converting the float point weights to fix point

the weight magnitudes are in [-2^{BIT_WIDTH_WEIGHT-1}, 2^{BIT_WIDTH_WEIGHT-1}-1]
the weight manitudes width of weight need to be e.g. 2 or 4 or 6 ... 32 and the
bit width for activation output is alike
"""
import keras
import numpy as np
import os
import ann_normalize
from copy import deepcopy
import data_preprocessing

BIT_WIDTH_WEIGHT_MIN = 2
BIT_WIDTH_WEIGHT_MAX = 32

fcn_norm_fix_model_path = os.path.join(
    ".", "ann_model_descs/fcn_normed_fixed_model.h5")
conv_norm_fix_model_path = os.path.join(
    ".", "ann_model_descs/conv_normed_fixed_model.h5")
conv_bn_norm_fix_model_path = os.path.join(
    ".", "ann_model_descs/conv_bn_normed_fixed_model.h5")


def load_model(model_type: int) -> keras.models.Model:
    if model_type == ann_normalize.TYPE_FCN:
        return keras.models.load_model(ann_normalize.fcn_norm_model_path)
    elif model_type == ann_normalize.TYPE_CONV_BN:
        return keras.models.load_model(ann_normalize.conv_norm_model_path)
    else:
        return keras.models.load_model(ann_normalize.conv_bn_norm_model_path)


def fix_point(model: keras.models.Model, bit_width_weight: int, dataX: np.array) -> keras.models.Model:
    if bit_width_weight < BIT_WIDTH_WEIGHT_MIN or bit_width_weight > BIT_WIDTH_WEIGHT_MAX:
        raise AssertionError(
            "Bit width is not valid, assume to be 2,4,6,8 or alike")

    beg_idx = 0
    if model.layers[0].__class__.__name__ == "InputLayer":
        beg_idx = 1

    scale_bottom = -(2**(bit_width_weight-1))
    scale_up = (2**(bit_width_weight-1))-1
    layer_weights = []
    layer_idxs = []
    for i in range(beg_idx, len(model.layers)):
        if len(np.shape(model.layers[i].get_weights())) <= 1:
            continue
        layer_weights.append(
            deepcopy(np.array(model.layers[i].get_weights(), dtype="float32")))
        layer_idxs.append(i)

    max_wt = max([np.max(wts) for wts in layer_weights])
    min_wt = min([np.min(wts[wts >= 0]) for wts in layer_weights])
    quant_factor = scale_up/(max_wt-min_wt)
    for i in range(len(layer_weights)):
        layer_weights[i] = np.floor(quant_factor*(layer_weights[i]-min_wt))

    max_wt_neg = min([np.min(wts[wts < 0]) for wts in layer_weights])
    min_wt_neg = max([np.max(wts[wts < 0]) for wts in layer_weights])
    quant_factor = scale_bottom/(max_wt_neg-min_wt_neg)
    for i in range(len(layer_weights)):
        layer_weights[i] = np.ceil(quant_factor*(layer_weights[i]-min_wt))

    for i in range(len(layer_idxs)):
        model.layers[layer_idxs[i]].set_weights(layer_weights[i])

    return model


def test_model(model: keras.models.Model, dataX: np.array, dataY: np.array):
    preds = model.predict(dataX, batch_size=1, verbose=1)
    preds = np.argmax(preds, axis=-1)
    accu = np.sum(preds == np.argmax(dataY, axis=-1))/len(preds)
    print(
        "After convert to fix point integer, accuracy is {:.2%}".format(accu))


def save_model(model: keras.models.Model, model_type: int):
    if model_type == ann_normalize.TYPE_FCN:
        model.save(fcn_norm_fix_model_path)
    elif model_type == ann_normalize.TYPE_CONV:
        model.save(conv_norm_fix_model_path)
    else:
        model.save(conv_bn_norm_fix_model_path)


if __name__ == "__main__":
    # test fcn network fix point
    # ----------------------------
    # model = load_model(ann_normalize.TYPE_FCN)
    # dataX, dataY = data_preprocessing.load_mnist_dataset(
    #     data_need_flatten=True)
    # fixed_point_model = fix_point(model, 8, dataX[-10:])
    # test_model(fixed_point_model, dataX[:100], dataY[:100])
    # save_model(fixed_point_model, ann_normalize.TYPE_FCN)

    # test conv neural network fix point
    # --------------------------------------
    # model = load_model(ann_normalize.TYPE_CONV)
    # dataX, dataY = data_preprocessing.load_mnist_dataset(
    #     data_need_flatten=False)
    # fixed_point_model = fix_point(model, 8, dataX[-10:])
    # test_model(fixed_point_model, dataX[-100:], dataY[-100:])
    # save_model(fixed_point_model, ann_normalize.TYPE_CONV)

    # test conv bn neural network fix point
    model = load_model(ann_normalize.TYPE_CONV_BN)
    dataX, dataY = data_preprocessing.load_mnist_dataset(
        data_need_flatten=False)
    fixed_point_model = fix_point(model, 8, dataX[-100:])
    test_model(fixed_point_model, dataX[:100], dataY[:100])
    save_model(fixed_point_model, ann_normalize.TYPE_CONV_BN)
