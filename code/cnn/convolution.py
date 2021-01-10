"""
Function：卷积
Author：lzb
Date：2021.01.08
"""

import numpy as np

from enum import Enum
from gl import errorcode


"""
class：CVLDim 卷积维度枚举值
TWO：2维
THREE：3维
"""


class CVLDim(Enum):
    TWO = 2
    THREE = 3


"""
class：Reversal 卷积翻转枚举值
REV：翻转
NO_REV：不翻转
"""


class Reversal(Enum):
    REV = 1
    NO_REV = 0


"""
class：ConvolutionType 卷积类型枚举值
Narrow：窄卷积，S = 1， 两端不补零 P = 0
Wide：宽卷积，S = 1，两端补零 P = K - 1
Equal_Width，等宽卷积，S = 1，两端补零 P = (K - 1) / 2
Other：其他（S、P，由参数决定）
"""


class ConvolutionType(Enum):
    Narrow = 0
    Wide = 1
    Equal_Width = 2
    Other = 3


"""
class：Convolution 卷积（二维卷积）
"""


class Convolution:
    # 卷积维度（是2维数组的卷积还是3维数组的卷积）
    cvl_dim = CVLDim.TWO.value

    # 卷积核
    w = 0

    # 卷积核 width, height, depth
    w_width = 0
    w_height = 0
    w_depth = 0

    # 卷积翻转标记
    rev = Reversal.REV

    # 步长
    s = 1

    # 补齐长度
    padding_row = 0
    padding_col = 0

    """
    功能：计算卷积
    参数：
    w：卷积核
    x：输入信息
    rev：卷积是否翻转（枚举值）
    con_type：卷积类型
    s：步长
    padding_row：矩阵的行，补零长度
    padding_col：矩阵的列，补零长度
    返回值：
    y：卷积结果(一个矩阵)
    err：错误码
    """

    def convolution(self, w, x, rev=Reversal.NO_REV, con_type=ConvolutionType.Narrow,
                    s=1, padding_row=0, padding_col=0):
        # 1. 合法性校验
        err = self._valid(w, x, s, padding_row, padding_col)

        if errorcode.SUCCESS != err:
            return 0, err  # 如果错误，卷积结果返回 0

        # 2. 成员变量赋值
        self.w = w
        self.rev = rev

        self.cvl_dim = len(w.shape)

        self.w_width = w.shape[0]
        self.w_height = w.shape[1]

        # 如果是3维卷积，则卷积核有 depth
        if CVLDim.THREE.value == self.cvl_dim:
            self.w_depth = w.shape[2]
        # 否则的话，depth = 0
        else:
            self.w_depth = 0

        # 3. 计算步长和补零长度
        self._cal_step_padding(con_type, s, padding_row, padding_col)

        # 4. 将 x 扩充（补零）
        if (padding_row > 0) or (padding_col > 0):
            x = self._padding(x)

        # 5. 翻转 w（如果需要的话）
        if Reversal.REV == rev:
            self._reverse_w()

        # 6. 计算卷积 y
        y = self._cal_convolution(x)

        # 7. 返回卷积计算结果和错误码
        err = errorcode.SUCCESS
        return y, err

    """
    功能：参数校验
    参数：
    w：滤波器（一个矩阵）
    x：输入信息（一个矩阵）    
    s：步长
    padding_row：矩阵的行，补零长度
    padding_col：矩阵的列，补零长度
    返回值：错误码
    """

    def _valid(self, w, x, s, padding_row, padding_col):
        """
        # 因为 w, x 必须是 matrix，s 必须是 int
        # 所以，参数类型合法性校验，是个比较烦人的事情
        # 难道每一个函数，都得判断参数的类型
        # 或者每一个函数，都得用 try/except 来处理
        # 我也很迷茫，暂时不管那么多了
        # 暂时不做参数参数类型合法性校验
        # 这里就假设 w, x 是二维矩阵，s 是个 int
        """

        # TODO：待实现
        self.w = self.w

        return errorcode.SUCCESS

    """
    功能：计算步长和补零长度
    参数：
    con_type：卷积类型
    s：步长
    padding_row：矩阵的行，补零长度
    padding_col：矩阵的列，补零长度
    返回值：错误码
    """

    def _cal_step_padding(self, con_type, s, padding_row, padding_col):
        # python 竟然没有 switch/case
        if ConvolutionType.Narrow == con_type:
            self.s = 1
            self.padding_row = 0
            self.padding_col = 0
        elif ConvolutionType.Wide == con_type:
            self.s = 1
            self.padding_row = self.w_width - 1
            self.padding_col = self.w_height - 1
        elif ConvolutionType.Equal_Width == con_type:
            self.s = 1
            self.padding_row = (self.w_width - 1) // 2  # 这里不管那么多了，向下取整
            self.padding_col = (self.w_height - 1) // 2  # 这里不管那么多了，向下取整
        else:
            self.s = s
            self.padding_row = padding_row
            self.padding_col = padding_col

    """
    功能：扩充（补零） x
    参数：
    x：待扩充的矩阵    
    返回值：扩充后的矩阵
    """

    def _padding(self, x):
        # x 的 row count， col count
        x_row_count = x.shape[0]
        x_col_count = x.shape[1]

        # 补零后的 xp，先赋初值为 0
        if CVLDim.THREE.value == self.cvl_dim:
            xp = np.zeros([(x_row_count + self.padding_row), (x_col_count + self.padding_col), self.w_depth])
        else:
            xp = np.zeros([(x_row_count + self.padding_row), (x_col_count + self.padding_col)])

        # xp 中间的值，与 x 相同
        for i in range(0, x_row_count):
            for j in range(0, x_col_count):
                if CVLDim.THREE.value == self.cvl_dim:
                    for d in range(0, self.w_depth):
                        xp[(i + self.padding_row), (j + self.padding_col)] = x[i, j]
                else:
                    xp[(i + self.padding_row), (j + self.padding_col)] = x[i, j]

        return xp

    """
    功能：翻转矩阵
    参数：NULL    
    返回值：翻转后的矩阵
    """

    def _reverse_w(self):
        half_w_width = self.w_width // 2  # 向下取整
        half_w_height = self.w_height // 2  # 向下取整

        for i in range(0, half_w_width + 1):
            for j in range(0, half_w_height + 1):
                if CVLDim.THREE.value == self.cvl_dim:
                    for d in range(0, self.w_depth):
                        tmp = self.w[i, j, d]
                        self.w[i, j, d] = self.w[(self.w_width - 1 - i), (self.w_height - 1 - j), d]
                        self.w[(self.w_width - 1 - i), (self.w_height - 1 - j), d] = tmp
                else:
                    tmp = self.w[i, j]
                    self.w[i, j] = self.w[(self.w_width - 1 - i), (self.w_height - 1 - j)]
                    self.w[(self.w_width - 1 - i), (self.w_height - 1 - j)] = tmp

    """
    功能：计算2维卷积
    参数：
    x：输入信息（一个矩阵）
    返回值：卷积结果
    """

    def _cal_convolution(self, x):
        # 卷积的 width，height
        y_width, y_height = cal_cvl_wh(self.w, x, self.s)

        # 初始化卷积 y

        # 如果是3维卷积
        if CVLDim.THREE.value == self.cvl_dim:
            y = np.zeros([y_width, y_height, self.w_depth])
        else:
            y = np.zeros([y_width, y_height])

        # 计算卷积 y
        for i in range(0, y_width):
            for j in range(0, y_height):
                for u in range(0, self.w_width):
                    for v in range(0, self.w_height):
                        # 3维卷积
                        if CVLDim.THREE.value == self.cvl_dim:
                            for d in range(0, self.w_depth):
                                y[i, j, d] += self.w[u, v, d] * x[(i * self.s + u), (j * self.s + v), d]
                        # 2维卷积
                        else:
                            y[i, j] += self.w[u, v] * x[(i * self.s + u), (j * self.s + v)]

        return y


"""
功能：计算卷积结果的矩阵的宽度和高度
参数：
w：卷积核（一个矩阵）
x：输入信息（一个矩阵）
s：步长
返回值：NULL
"""


def cal_cvl_wh(w, x, s):
    # w, x 的 width, height
    w_width = w.shape[0]
    w_height = w.shape[1]
    x_width = x.shape[0]
    x_height = x.shape[1]

    # 卷积的 row count 和 col count
    width = (x_width - w_width) // s + 1  # 向下取整
    height = (x_height - w_height) // s + 1  # 向下取整

    return width, height


