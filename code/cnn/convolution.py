"""
Function：卷积
Author：lzb
Date：2021.01.08
"""

import numpy as np

from enum import Enum
from gl import errorcode

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
    # 滤波器
    w = 0

    # 滤波器 row count，col count
    w_row_count = 0
    w_col_count = 0

    # 卷积翻转标记
    rev = Reversal.REV

    # 步长
    s = 1

    # 补齐长度
    padding_row = 0
    padding_col = 0

    """
    功能：计算二维卷积
    参数：
    w：滤波器（一个矩阵）
    x：输入信息（一个矩阵）
    rev：卷积是否翻转（枚举值）
    con_type：卷积类型
    s：步长
    padding_row：矩阵的行，补零长度
    padding_col：矩阵的列，补零长度
    返回值：
    y：卷积结果(一个矩阵)
    err：错误码
    """

    def convolution_d2(self, w, x, rev=Reversal.REV, con_type=ConvolutionType.Narrow,
                       s=1, padding_row=0, padding_col=0):
        # 1. 合法性校验
        err = self._valid(w, x, s, padding_row, padding_col)

        if errorcode.SUCCESS != err:
            return 0, err  # 如果错误，卷积结果返回 0

        # 2. 成员变量赋值
        self.w = w
        self.rev = rev

        self.w_row_count = w.shape[0]
        self.w_col_count = w.shape[1]

        # 3. 计算步长和补零长度
        self._cal_step_padding(con_type, s, padding_row, padding_col)

        # 4. 将 x 扩充（补零）
        if (padding_row > 0) or (padding_col > 0):
            x = self._padding(x)

        # 5. 翻转 w（如果需要的话）
        if Reversal.REV == rev:
            self._reverse_w()

        # 6. 计算卷积 y
        y = self._cal_convolution_d2(x)

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
            self.padding_row = self.w_row_count - 1
            self.padding_col = self.w_col_count - 1
        elif ConvolutionType.Equal_Width == con_type:
            self.s = 1
            self.padding_row = (self.w_row_count - 1) // 2  # 这里不管那么多了，向下取整
            self.padding_col = (self.w_col_count - 1) // 2  # 这里不管那么多了，向下取整
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
        xp = np.zeros([(x_row_count + self.padding_row), (x_col_count + self.padding_col)])

        # xp 中间的值，与 x 相同
        for i in range(0, x_row_count):
            for j in range(0, x_col_count):
                xp[(i + self.padding_row), (j + self.padding_col)] = x[i, j]

        return xp

    """
    功能：翻转矩阵
    参数：NULL    
    返回值：翻转后的矩阵
    """

    def _reverse_w(self):
        half_w_row_count = self.w_row_count // 2  # 向下取整
        half_w_col_count = self.w_col_count // 2  # 向下取整

        for i in range(0, half_w_row_count + 1):
            for j in range(0, half_w_col_count + 1):
                tmp = self.w[i, j]
                self.w[i, j] = self.w[(self.w_row_count - 1 - i), (self.w_col_count - 1 - j)]
                self.w[(self.w_row_count - 1 - i), (self.w_col_count - 1 - j)] = tmp

    """
    功能：计算卷积
    参数：
    x：输入信息（一个矩阵）
    返回值：卷积结果
    """

    def _cal_convolution_d2(self, x):
        # x 的 row count， col count
        x_row_count = x.shape[0]
        x_col_count = x.shape[1]

        # 卷积的 row count 和 col count
        y_row_count = (x_row_count - self.w_row_count + 1) // self.s  # 向下取整
        y_col_count = (x_col_count - self.w_col_count + 1) // self.s  # 向下取整

        # 初始化卷积 y
        y = np.zeros([y_row_count, y_col_count])

        # 计算卷积 y
        for i in range(0, y_row_count):
            for j in range(0, y_col_count):
                for u in range(0, self.w_row_count):
                    for v in range(0, self.w_col_count):
                        y[i, j] += self.w[u, v] * x[(i * self.s + u), (j * self.s + v)]

        return y


"""
test
"""


def test_convolution():
    x = np.asarray([[1, 1, 1, 1, 1],
                    [-1, 0, -3, 0, 1],
                    [2, 1, 1, -1, 0],
                    [0, -1, 1, 2, 1],
                    [1, 2, 1, 1, 1]])

    w = np.asarray([[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, -1]])

    con = Convolution()

    y, err = con.convolution_d2(w, x)

    print("\n")

    print(y)
