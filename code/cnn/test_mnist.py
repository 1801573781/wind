"""
Function：test mnist
Author：lzb
Date：2021.01.25
说明：经过了几天的崩溃以后，先换个角度。先从 mnist 数据集使用开始
"""

from keras.datasets import mnist


def test():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  ' + str(test_X.shape))
    print('Y_test:  ' + str(test_y.shape))
