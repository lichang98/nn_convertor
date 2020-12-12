# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import os
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

mnist_cate_dir = os.path.join("..", "dataset/trainingSet")


def load_mnist_dataset(data_need_flatten=True) -> Tuple[np.array, np.array]:
    """
    Load Mnist dataset
    The data is in image format with 28*28 pixels
    The images are classified by their labels and in corresponding directory
    e.g. dir-
            |
            ---0
                |
                ----img0001.jpg
                ----img0002.jpg
            ----1
                |
                ----img1001.jpg
                ----img1002.jpg
            ......
    Images will be loaded into numpy array with dtype float32, and normalized into
    range [0,1]
    @param data_need_flatten: whether the image are keep two dimsion or flatten to one dimension
    @return:
            dataX: M*N dim array floa32, where M is number of samples, N is the total number of pixels of one image
            dataY: M dim array int32, M is number of samples, labels for correspding image
    """
    dataX, dataY = [], []
    for subdir in os.listdir(mnist_cate_dir):
        for img_file in os.listdir(os.path.join(mnist_cate_dir, subdir)):
            img = Image.open(os.path.join(mnist_cate_dir, subdir, img_file))
            img = np.array(img, dtype="float32")
            img /= 255.0
            if data_need_flatten is True:
                dataX.append(img.flatten())
            else:
                if len(np.shape(img)) < 3:
                    dataX.append(np.expand_dims(img,axis=-1))
            dataY.append(int(subdir))

    dataX = np.array(dataX, dtype="float32")
    dataY = np.expand_dims(np.array(dataY, dtype="int32"), axis=-1)
    onehot_enc = OneHotEncoder()
    onehot_enc.fit(dataY)
    dataY = np.array(onehot_enc.transform(dataY).toarray(), dtype="int32")
    dataX, dataY = shuffle(dataX, dataY)
    return dataX, dataY


if __name__ == "__main__":
    dataX, dataY = load_mnist_dataset()
    print(dataX.shape)
    print(dataY.shape)
