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
    ann_normalize.test_ann(model,dataX[:100],dataY[:100])
    print("Start parameter change...")
    model_norm = ann_normalize.param_normalization(model,dataX[-100:])
    fix_point_model = ann_fix_point.fix_point(model_norm,16,dataX[-100:])
    print("Model Parameter change finish!")
    print("Start Test model after change...")
    accu = ann_fix_point.test_model(fix_point_model,dataX[:100], dataY[:100])
    print("After model accuracy is {:.2%}".format(accu))
    print("Test done!")

    