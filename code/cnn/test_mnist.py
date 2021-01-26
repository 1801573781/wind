"""
Function：test mnist
Author：lzb
Date：2021.01.25
说明：经过了几天的崩溃以后，先换个角度。先从 mnist 数据集使用开始
"""

from keras.datasets import mnist
from matplotlib import pyplot

import tensorflow as tf


def test1():
    print("\nhello\n")

    # print(tf.test.is_gpu_available())
    tf.config.list_physical_devices('GPU')

    print("\ngpu\n")

    # 保证sess.run()能够正常运行
    tf.compat.v1.disable_eager_execution()
    hello = tf.constant('hello,tensorflow')
    # 版本2.0的函数
    sess = tf.compat.v1.Session()
    print(sess.run(hello))

    print("\nagain\n")


def test2():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  ' + str(test_X.shape))
    print('Y_test:  ' + str(test_y.shape))

    for i in range(0, 20):
        # pyplot.subplot(330 + 1 + i)
        pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
        pyplot.show()

