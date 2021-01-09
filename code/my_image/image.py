"""
Function：图像处理
Author：lzb
Date：2021.01.09
"""

import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

from enum import Enum

from gl import errorcode

"""
class：ImageDataType，图像数据类型枚举值
"""


class ImageDataType(Enum):
    R = 0
    G = 1
    B = 2
    RGB = 3
    RGBA = 4
    GRAY = 5
    OTHER = 6
    ERROR = 7


"""
class：RGBComponent，RGB 分量枚举值
"""


class RGBComponent(Enum):
    R = 0
    G = 1
    B = 2


"""
功能：读取一个图像文件的图像数据
参数：
file_name：图像文件名
返回值：
data：图像数据
image_data_type：RGBA or RGB or GRAY
err：错误码
"""


def get_data(file_name):
    # 1. 读取图像
    # 如果出异常了，那就是文件不存在或者不是图像文件
    try:
        data = mpimg.imread(file_name)
    except BaseException as err_msg:
        print(err_msg)
        return 0, ImageDataType.ERROR, errorcode.FAILED

    # 2. 判断图像类型
    shape = data.shape

    # 如果 shape 是2维，那么图像数据类型是 GRAY
    if 2 == len(shape):
        image_data_type = ImageDataType.GRAY

    # 如果 shape 是3维，那么需要判断第3维的特征
    elif 3 == len(shape):
        t = shape[2]

        # 如果 t 是"一元组"，那么图像数据类型是 GRAY（这个判断可能有点问题，不过这里就这么处理了）
        if 1 == t:
            image_data_type = ImageDataType.GRAY
        # 如果 t 是"三元组"，那么图像数据类型是 RGB
        elif 3 == t:
            image_data_type = ImageDataType.RGB
        # 如果 t 是"四元组"，那么图像数据类型是 RGBA（也有可能是 ARGB，不过这里不管这么多了）
        elif 4 == t:
            image_data_type = ImageDataType.RGB
        else:
            image_data_type = ImageDataType.ERROR

    # 如果 shape 既不是2维，也不是3维，那就是 ERROR
    else:
        image_data_type = ImageDataType.ERROR

    return data, image_data_type, errorcode.SUCCESS


"""
功能：将一个 RGB 图像转为灰度图像
参数：NULL    
返回值：
gray：图像的灰度数组（是一个2维数组）
err：错误码
"""


def gray_file(file_name):
    # 1. 读取图像
    data, image_data_type, err = get_data(file_name)

    if errorcode.SUCCESS != err:
        return err

    if ImageDataType.ERROR == image_data_type:
        return errorcode.FAILED

    # 2. 将 data 转换为 gray

    if ImageDataType.GRAY == image_data_type:
        return data, errorcode.SUCCESS
    else:
        return gray_data(data), errorcode.SUCCESS


"""
功能：将一个 RGB 图像转为灰度图像
参数：
rgb_data：一个图像的 rgb 数据（3维数组）    
返回值：
gray：图像的灰度数组（是一个2维数组）
特别说明：这里忽略 RGBA or ARGB（以后再补上）
"""


def gray_data(rgb_data):
    # 图像宽度和高度
    width = rgb_data.shape[0]
    height = rgb_data.shape[1]

    # 构建灰度图像2维数组
    gray = np.zeros([width, height])

    # 转换成灰度
    for i in range(0, width):
        for j in range(0, height):
            # rgb 是一个3元组
            rgb = rgb_data[i, j]

            # 灰度转换公式为：0.299 * R + 0.587 * G + 0.114 * B
            gray[i, j] = 0.299 * rgb[RGBComponent.R.value] + \
                         0.587 * rgb[RGBComponent.G.value] + \
                         0.114 * rgb[RGBComponent.B.value]

    return gray


"""
功能：获取一个 RGB 图像的 R 分量
参数：
rgb_data：一个图像的 rgb 数据（3维数组）    
返回值：图像的 R 分量数组（是一个2维数组）
"""


def r_data(rgb_data):
    return _rgb_component_data(rgb_data, RGBComponent.R)


"""
功能：获取一个 RGB 图像的 G 分量
参数：
rgb_data：一个图像的 rgb 数据（3维数组）    
返回值：图像的 R 分量数组（是一个2维数组）
"""


def g_data(rgb_data):
    return _rgb_component_data(rgb_data, RGBComponent.G)


"""
功能：获取一个 RGB 图像的 B 分量
参数：
rgb_data：一个图像的 rgb 数据（3维数组）    
返回值：图像的 B 分量数组（是一个2维数组）
"""


def b_data(rgb_data):
    return _rgb_component_data(rgb_data, RGBComponent.B)


"""
功能：获取一个 RGB 图像的 R 分量
参数：
file_name：图像名称    
返回值：
data：图像的 R 分量数组（是一个2维数组）
err：错误码
"""


def r_component_file(file_name):
    return rgb_component_file(file_name, RGBComponent.R)


"""
功能：获取一个 RGB 图像的 G 分量
参数：
file_name：图像名称    
返回值：
data：图像的 G 分量数组（是一个2维数组）
err：错误码
"""


def g_component_file(file_name):
    return rgb_component_file(file_name, RGBComponent.G)


"""
功能：获取一个 RGB 图像的 B 分量
参数：
file_name：图像名称    
返回值：
data：图像的 B 分量数组（是一个2维数组）
err：错误码
"""


def b_component_file(file_name):
    return rgb_component_file(file_name, RGBComponent.B)


"""
功能：获取一个 RGB 图像的 RGB 分量
参数：
file_name：图像名称    
返回值：
data：图像的 RGB 分量数组（是一个2维数组）
err：错误码
"""


def rgb_component_file(file_name, rgb_component):
    # 1. 读取图像
    data, image_data_type, err = get_data(file_name)

    if errorcode.SUCCESS != err:
        return err

    if ImageDataType.ERROR == image_data_type:
        return errorcode.FAILED

    # 2. 获取 RGB 分量

    if ImageDataType.GRAY == image_data_type:
        return 0, errorcode.FAILED
    else:
        return _rgb_component_data(data, rgb_component), errorcode.SUCCESS


"""
功能：获取一个 RGB 图像的 RGB 分量
参数：
rgb_data：一个图像的 rgb 数据（3维数组）    
返回值：图像的 RGB 分量数组（是一个2维数组）
特别说明：这里忽略 RGBA or ARGB（以后再补上）
"""


def _rgb_component_data(rgb_data, rgb_component):
    # 图像宽度和高度
    width = rgb_data.shape[0]
    height = rgb_data.shape[1]

    # 构建分量图像2维数组
    data = np.zeros([width, height])

    # 提取分量
    for i in range(0, width):
        for j in range(0, height):
            # rgb 是一个3元组
            rgb = rgb_data[i, j]

            # 提取 RGB 分量
            data[i, j] = rgb[rgb_component.value]

    return data


"""
功能：显示一个图像（文件名）
参数：
file_name：图像文件名    
返回值：错误码
"""


def show_file(file_name):
    # 1. 读取图像
    data, image_data_type, err = get_data(file_name)

    if errorcode.SUCCESS != err:
        return err

    if ImageDataType.ERROR == image_data_type:
        return errorcode.FAILED

    # 显示图像
    show_data(data, image_data_type)


"""
功能：显示一个图像（图像数据）
参数：
data：图像数据    
image_data_type：图像数据类型
返回值：错误码
"""


def show_data(data, image_data_type):
    if ImageDataType.R == image_data_type:
        cmap = "Reds"
    elif ImageDataType.G == image_data_type:
        cmap = "Greens"
    elif ImageDataType.B == image_data_type:
        cmap = "Blues"
    elif ImageDataType.RGB == image_data_type:
        cmap = None
    elif ImageDataType.RGBA == image_data_type:
        cmap = None
    elif ImageDataType.OTHER == image_data_type:
        cmap = None
    elif ImageDataType.GRAY == image_data_type:
        cmap = "Greys_r"
    else:
        cmap = None

    plt.imshow(data, cmap)

    """
    # 如果灰度图像，就灰度显示
    if ImageDataType.GRAY == image_data_type:
        # plt.imshow(data, cmap='Greys_r')
        # plt.imshow(data, cmap="Blues")
        plt.imshow(data, cmap="Reds")
    # 否则，就正常显示
    else:
        plt.imshow(data)
    """

    plt.axis('off')  # 不显示坐标轴
    plt.show()


"""
test
"""


def test():
    file_name = "./dog1.bmp"

    show_file(file_name)

    data, err = gray_file(file_name)
    show_data(data, ImageDataType.GRAY)

    data, err = r_component_file(file_name)
    show_data(data, ImageDataType.R)

    data, err = g_component_file(file_name)
    show_data(data, ImageDataType.G)

    data, err = g_component_file(file_name)
    show_data(data, ImageDataType.B)
