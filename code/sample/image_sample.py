"""
Function：构建猫、狗的样本（灰度图）
Author：lzb
Date：2020.12.25
"""

import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

"""
x = np.linspace(0, 20 * np.pi, 300)
y = 10 * np.sin(x)
plt.plot(x, y, color='blue', linewidth=1.0)
plt.show()
"""


def test():
    lena = mpimg.imread('./../my_image/dog1.png')  # 读取和代码处于同一目录下的 lena.png
    # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
    shape = lena.shape

    plt.imshow(lena)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    # 显示图片的第一个通道
    lena_1 = lena[:, :, 2]
    plt.imshow(lena_1)
    plt.show()

    # 此时会发现显示的是热量图，不是我们预想的灰度图，可以添加 cmap 参数，有如下几种添加方法：
    plt.imshow(lena_1, cmap='Greys_r')
    plt.show()

    img = plt.imshow(lena_1)
    img.set_cmap('gray')  # 'hot' 是热量图
    plt.show()
