# -*- coding:utf-8 -*-
"""
This file is for integrating into darwin ide which powered by vscode
"""
import pickle
import sys
import keras
import ann_normalize
import ann_fix_point

if __name__ == "__main__":
    with open("stage1_tmp.pkl","rb") as f:
        dataX,dataY = pickle.load(f)
    
    model_path = sys.argv[1]
    print("Loading ANN model")
    model = keras.models.load_model(model_path)
    print("Loading done!")
    print("Test original ANN model....")
    accu = ann_normalize.test_ann(model,dataX[-200:],dataY[-200:])
    print("Original ANN model accuracy={:.2%}".format(accu))
    print("Start parameter change...")
    model_norm = ann_normalize.param_normalization(model,dataX[:10000])
    ann_normalize.save_normed_model(model_norm, ann_normalize.TYPE_FCN)
    fix_point_model = ann_fix_point.fix_point(model_norm,16,dataX[:10000])
    ann_fix_point.save_model(fix_point_model,ann_normalize.TYPE_FCN)
    print("Model Parameter change finish!")
    print("Start Test model after change...")
    accu = ann_fix_point.test_model(fix_point_model,dataX[-200:], dataY[-200:])
    print("After model accuracy is {:.2%}".format(accu))
    print("Test done!")

    