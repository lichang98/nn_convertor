# -*- coding:utf-8 -*-
"""
This file is for integrating into darwin ide which powered by vscode
"""
import sys
import pickle
import numpy as np
import data_preprocessing

if __name__ == "__main__":
    data_dir = sys.argv[1] # where test data placed
    print("given data dir path: "+data_dir)
    data_preprocessing.mnist_cate_dir = data_dir
    dataX,dataY = data_preprocessing.load_mnist_dataset()
    print("processed datax shape={}, datay shape={}".format(np.shape(dataX),np.shape(dataY)))
    with open("stage1_tmp.pkl","wb") as f:
        pickle.dump((dataX,dataY), f)
    