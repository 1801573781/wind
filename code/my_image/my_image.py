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
from gl.common_enum import ArrayDim

# 定义一个极小值
TINY = 0.001


class ImageDataType(Enum):
    """
    图像数据类型枚举值
    """

    R = 0
    G = 1
    B = 2
    RGB = 3
    RGBA = 4
    GRAY = 5
    OTHER = 6
    ERROR = 7


class RGBComponent(Enum):
    """
    RGB 分量枚举值
    """
    R = 0
    G = 1
    B = 2


class NormalizationType(Enum):
    """
    图像数据归一化类型\n
    NORMAL：将图像数据归一化到 0~1\n
    UN_NORMAL：将图像数据反归一化到 0~255\n
    """

    NORMAL = 1
    REV_NORMAL = 2


def get_data(file_name):
    """
    功能：读取一个图像文件的图像数据\n
    参数：\n
    file_name：图像文件名\n
    返回值：\n
    data：图像数据\n
    image_data_type：RGBA or RGB or GRAY\n
    err：错误码\n
    """

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


def gray_file(file_name, dim=ArrayDim.TWO):
    """
    功能：将一个 RGB 图像转为灰度图像\n
    参数：\n
    file_name：图像文件名\n
    dim：指明返回值是2维数组还是3维数组\n
    返回值：\n
    gray：图像的灰度数组（是一个2维数组）\n
    err：错误码\n
    """

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
        return gray_data(data, dim), errorcode.SUCCESS


def gray_data(rgb_data, dim=ArrayDim.TWO):
    """
    功能：将一个 RGB 图像转为灰度图像\n
    参数：\n
    rgb_data：一个图像的 rgb 数据（3维数组）\n
    返回值：\n
    gray：图像的灰度数组（是一个2维数组 or 3维数组）\n
    dim：指明返回值是2维数组还是3维数组\n
    特别说明：这里忽略 RGBA or ARGB（以后再补上）\n
    """

    # 图像宽度和高度
    width = rgb_data.shape[0]
    height = rgb_data.shape[1]

    # 构建灰度图像2维数组
    if ArrayDim.TWO == dim:
        gray = np.zeros([width, height])
    # 构建灰度图像3维数组
    else:
        gray = np.zeros([width, height, 1])

    # 转换成灰度
    for i in range(0, width):
        for j in range(0, height):
            # rgb 是一个3元组
            rgb = rgb_data[i, j]

            # 灰度转换公式为：0.299 * R + 0.587 * G + 0.114 * B
            # 灰度图像2维数组
            if ArrayDim.TWO == dim:
                gray[i, j] = 0.299 * rgb[RGBComponent.R.value] + \
                             0.587 * rgb[RGBComponent.G.value] + \
                             0.114 * rgb[RGBComponent.B.value]
            # 灰度图像3维数组
            else:
                gray[i, j, 0] = 0.299 * rgb[RGBComponent.R.value] + \
                                0.587 * rgb[RGBComponent.G.value] + \
                                0.114 * rgb[RGBComponent.B.value]

    return gray


def r_data(rgb_data, dim=ArrayDim.TWO):
    """
    功能：获取一个 RGB 图像的 R 分量\n
    参数：\n
    rgb_data：一个图像的 rgb 数据（3维数组）\n
    dim：指明返回值是2维数组还是3维数组\n
    返回值：图像的 R 分量数组（是一个2维数组 or 3维数组）\n
    """

    return _rgb_component_data(rgb_data, RGBComponent.R, dim)


def g_data(rgb_data, dim=ArrayDim.TWO):
    """
    功能：获取一个 RGB 图像的 G 分量\n
    参数：\n
    rgb_data：一个图像的 rgb 数据（3维数组）\n
    dim：指明返回值是2维数组还是3维数组\n
    返回值：图像的 R 分量数组（是一个2维数组 or 3维数组）\n
    """

    return _rgb_component_data(rgb_data, RGBComponent.G, dim)


def b_data(rgb_data, dim=ArrayDim.TWO):
    """
    功能：获取一个 RGB 图像的 B 分量\n
    参数：\n
    rgb_data：一个图像的 rgb 数据（3维数组）\n
    dim：指明返回值是2维数组还是3维数组\n
    返回值：图像的 B 分量数组（是一个2维数组 or 3维数组）\n
    """

    return _rgb_component_data(rgb_data, RGBComponent.B, dim)


def r_component_file(file_name, dim=ArrayDim.TWO):
    """
    功能：获取一个 RGB 图像的 R 分量\n
    参数：\n
    file_name：图像名称\n
    dim：指明返回值是2维数组还是3维数组\n
    返回值：\n
    data：图像的 R 分量数组（是一个2维数组 or 3维数组）\n
    err：错误码\n
    """

    return rgb_component_file(file_name, RGBComponent.R, dim)


def g_component_file(file_name, dim=ArrayDim.TWO):
    """
    功能：获取一个 RGB 图像的 G 分量\n
    参数：\n
    file_name：图像名称\n
    dim：指明返回值是2维数组还是3维数组\n
    返回值：\n
    data：图像的 G 分量数组（是一个2维数组 or 3维数组）\n
    err：错误码\n
    """

    return rgb_component_file(file_name, RGBComponent.G, dim)


def b_component_file(file_name, dim=ArrayDim.TWO):
    """
    功能：获取一个 RGB 图像的 B 分量\n
    参数：\n
    file_name：图像名称\n
    dim：指明返回值是2维数组还是3维数组\n
    返回值：\n
    data：图像的 B 分量数组（是一个2维数组 or 3维数组）\n
    err：错误码\n
    """

    return rgb_component_file(file_name, RGBComponent.B, dim)


def rgb_component_file(file_name, rgb_component, dim=ArrayDim.TWO):
    """
    功能：获取一个 RGB 图像的 RGB 分量\n
    参数：\n
    file_name：图像名称\n
    dim：指明返回值是2维数组还是3维数组\n
    返回值：\n
    data：图像的 RGB 分量数组（是一个2维数组 or 3维数组）\n
    err：错误码\n
    """

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
        return _rgb_component_data(data, rgb_component, dim), errorcode.SUCCESS


def _rgb_component_data(rgb_data, rgb_component, dim=ArrayDim.TWO):
    """
    功能：获取一个 RGB 图像的 RGB 分量\n
    参数：\n
    rgb_data：一个图像的 rgb 数据（3维数组）\n
    dim：指明返回值是2维数组还是3维数组\n
    返回值：图像的 RGB 分量数组（是一个2维数组 or 3维数组）\n
    特别说明：这里忽略 RGBA or ARGB（以后再补上）\n
    """

    # 图像宽度和高度
    width = rgb_data.shape[0]
    height = rgb_data.shape[1]

    # 构建分量图像2维数组
    if ArrayDim.TWO == dim:
        data = np.zeros([width, height])
    # 构建分量图像3维数组
    else:
        data = np.zeros([width, height, 1])

    # 提取分量
    for i in range(0, width):
        for j in range(0, height):
            # rgb 是一个3元组
            rgb = rgb_data[i, j]

            # 提取 RGB 分量
            # 2维数组
            if ArrayDim.TWO == dim:
                data[i, j] = rgb[rgb_component.value]
            # 3维数组
            else:
                data[i, j, 0] = rgb[rgb_component.value]

    return data


def show_file(file_name):
    """
    功能：显示一个图像（文件名）\n
    参数：\n
    file_name：图像文件名\n
    返回值：错误码\n
    """

    # 1. 读取图像
    data, image_data_type, err = get_data(file_name)

    if errorcode.SUCCESS != err:
        return err

    if ImageDataType.ERROR == image_data_type:
        return errorcode.FAILED

    # 显示图像
    show_data(data, image_data_type)


def normalize(data, normalize_type=NormalizationType.NORMAL):
    """
    功能：将图像数据归一化（归一为 0~1） or 反归一化（反归一化为 0~255）\n
    参数：\n
    data：图像数据\n
    type：归一化类型\n
    返回值：图像数据归一化/反归一化的结果\n
    """

    # 数组维度
    shape = data.shape

    # 数组 width, height, depth
    width = shape[0]
    height = shape[1]

    if ArrayDim.TWO.value == len(shape):
        dim = ArrayDim.TWO
        depth = -1
        y = np.zeros([width, height])
    else:
        dim = ArrayDim.THREE
        depth = shape[2]
        y = np.zeros([width, height, depth])

    # 归一化
    if NormalizationType.NORMAL == normalize_type:
        _normalize(y, data, width, height, depth, dim)
    # 反归一化
    else:
        _rev_normalize(y, data, width, height, depth, dim)

    return y


def _normalize(y, data, width, height, depth, dim):
    """
    功能：将图像数据归一化（从 0~255 归一为 0~1）\n
    参数：\n
    y: 图像数据归一化的结果\n
    data：图像数据\n
    width：图像 width\n
    height：图像 height\n
    depth：图像 depth\n
    dim：图像数据维度（2维 or 3维数组）\n
    返回值：NULL\n
    """

    # 2维数组
    if ArrayDim.TWO == dim:
        for i in range(0, width):
            for j in range(0, height):
                y[i, j] = min(data[i, j] / 255, 1)
    # 3维数组
    else:
        for k in range(0, depth):
            for i in range(0, width):
                for j in range(0, height):
                    y[i, j, k] = min(data[i, j, k] / 255, 1)


def _rev_normalize(y, data, width, height, depth, dim):
    """
    功能：将图像数据反归一化（从 0~1 反归一为 0~255）\n
    参数： \n
    y: 图像数据归一化的结果 \n
    data：图像数据 \n
    width：图像 width \n
    height：图像 height \n
    depth：图像 depth \n
    dim：图像数据维度（2维 or 3维数组）\n
    返回值：NULL \n
    """

    # 1. 先求最大值
    tmp_list = list()

    # 2维数组
    if ArrayDim.TWO == dim:
        for i in range(0, width):
            for j in range(0, height):
                tmp_list.append(data[i, j])
    # 3维数组
    else:
        for k in range(0, depth):
            for i in range(0, width):
                for j in range(0, height):
                    tmp_list.append(data[i, j, k])

    # 最大值
    tmp_max = max(tmp_list)

    # 2. 反归一化

    # 2维数组
    if ArrayDim.TWO == dim:
        for i in range(0, width):
            for j in range(0, height):
                y[i, j] = data[i, j] / tmp_max * 255
    # 3维数组
    else:
        for k in range(0, depth):
            for i in range(0, width):
                for j in range(0, height):
                    y[i, j, k] = data[i, j, k] / tmp_max * 255


def array_3_2(data):
    """
    功能：将3维数组转换成2维数组 \n
    参数： \n
    data：图像数据，3维数组 \n
    返回值：图像数据，2维数组 \n
    """

    # 受不了了，不判断那么多了
    shape = data.shape

    width = shape[0]
    height = shape[1]

    z = np.zeros([width, height])

    for i in range(0, width):
        for j in range(0, height):
            z[i, j] = data[i, j, 0]

    return z


def array_3_1(data):
    """
    功能：将3维数组转换成1维数组 \n
    参数： \n
    data：图像数据，3维数组 \n
    返回值：图像数据，1维数组 \n
    """

    # 受不了了，不判断那么多了
    shape = data.shape

    width = shape[0]
    height = shape[1]

    # z = list()

    z = np.zeros([width * height, 1])

    for i in range(0, width):
        for j in range(0, height):
            index = i * width + j
            z[index][0] = data[i, j, 0]
            # z.append(data[i, j, 0])

    return z


def array_0_tiny(data):
    """
    功能：将数组中的 0，转化为极小值 \n
    参数： \n
    data：图像数据，2维 or 3维数组 \n
    返回值：NULL \n

    说明：对于图像数据而言，如果像素点是黑色，其 value = 0 （or [0, 0, 0]）， \n
    这造成有些运算中，会出现分母为0的情况。所以，这里把 0 转换为一个极小的值 \n

    """

    shape = data.shape

    if ArrayDim.THREE.value == len(shape):
        dim = ArrayDim.THREE
        depth = shape[2]
    else:
        dim = ArrayDim.TWO
        depth = -1

    width = shape[0]
    height = shape[1]

    # 3维数组
    if ArrayDim.THREE == dim:
        for k in range(0, depth):
            for i in range(0, width):
                for j in range(0, height):
                    # 由于是 float 类型，不能直接用 "== 0" 来判断，小于1个极小的数，就认为是0
                    if data[i, j, k] < TINY:
                        data[i, j, k] = TINY
    # 2维数组
    else:
        for i in range(0, width):
            for j in range(0, height):
                # 由于是 float 类型，不能直接用 "== 0" 来判断，小于1个极小的数，就认为是0
                if data[i, j] < TINY:
                    data[i, j] = TINY


def show_data(data, image_data_type):
    """
    功能：显示一个图像（图像数据） \n
    参数： \n
    data：图像数据 \n
    image_data_type：图像数据类型 \n
    返回值：错误码 \n
    """

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

    plt.axis('off')  # 不显示坐标轴

    plt.show()


"""
test
"""


def test():
    file_name = "../picture/base_test/dog1.bmp"

    show_file(file_name)

    data, err = gray_file(file_name)
    show_data(data, ImageDataType.GRAY)

    data, err = r_component_file(file_name)
    show_data(data, ImageDataType.R)

    data, err = g_component_file(file_name)
    show_data(data, ImageDataType.G)

    data, err = g_component_file(file_name)
    show_data(data, ImageDataType.B)
