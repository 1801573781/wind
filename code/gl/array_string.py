"""
Function：数组转换为字符串
Author：lzb
Date：2021.01.19

说明：
1、只处理1维、2维、3维数组
2、3个维度，在日常用语中的称呼：行、列、进（四合院，几进的院落）
"""
from gl.common_enum import ArrayDim

"""
功能：数组转换为字符串
参数：
arr：待转换为字符串的数组  
返回值：数组所转换的字符串
"""


def array_2_string(arr):
    # 数组的维度
    dim = len(arr.shape)

    # 1维数组
    if ArrayDim.ONE.value == dim:
        arr_2_str = D1ArrayString()
    # 2维数组
    elif ArrayDim.TWO.value == dim:
        arr_2_str = D2ArrayString()
    # 3维数组（剩下的都当作3维数组，超过3维的，暂时也当作3维数组）
    else:
        arr_2_str = D3ArrayString()

    return arr_2_str.array_2_string(arr)


"""
class：Array2String 数组转换为字符串(base class)
"""


class ArrayString:
    # 数组
    arr = None

    # 起始符
    start_char = "["

    # 终止符
    end_char = "]"

    # 分割符
    split_char = ", "

    # 换行符
    br = "\n"

    """
    功能：数组转换为字符串
    参数：
    arr：待转换为字符串的数组  
    返回值：数组所转换的字符串
    """

    def array_2_string(self, arr):
        # 不做合法性校验了，受不了了，太麻烦了
        self. arr = arr

        # 数组的宽度
        width = self.arr.shape[0]

        # 字符串初始化：整体起始符
        arr_str = self.start_char

        for row in range(0, width):
            # 每一行的起始符
            arr_str += self._get_row_start_char()

            # 每一行的数据所转换的字符串
            arr_str += self._get_row_string(row)

            # 每一行的终止符
            arr_str += self._get_row_end_char()

            # 行与行之间的分隔符与换行符
            if row < (width - 1):
                arr_str += self.split_char  # 分隔符
                arr_str += self._get_row_br_char()   # 换行符

        # 整体终止符
        arr_str += self.end_char

        return arr_str

    """
    功能：获取 每一个 row 的起始符
    参数：NULL           
    返回值：每一个 row 的起始符
    """

    def _get_row_start_char(self):
        return self.start_char

    """
    功能：获取 每一个 row 的终止符
    参数：NULL           
    返回值：每一个 row 的终止符
    """

    def _get_row_end_char(self):
        return self.end_char

    """
    功能：获取 每一个 row 的换行符
    参数：NULL           
    返回值：每一个 row 的换行符
    """

    def _get_row_br_char(self):
        return self.br

    """
    功能：获取分割符（列之间，进之间）
    参数：NULL           
    返回值：每一个 row 的换行符
    """

    def _get_split_char(self):
        return self.split_char

    """
    功能：获取行数据字符串
    参数：
    row：行号
    返回值：某一行的数据的字符串
    """

    def _get_row_string(self, row):
        pass

    """
    功能：获取列数据字符串
    参数：
    row：行号
    col：列号
    返回值：某一列的数据的字符串
    """

    def _get_col_string(self, row, col):
        pass

    """
    功能：获取进数据字符串
    参数：
    row：行号
    col：列号
    kin: 进号
    返回值：某一列的数据的字符串
    """

    def _get_kin_string(self, row, col, kin):
        pass


"""
class：D1ArrayString 1维数组转换为字符串
"""


class D1ArrayString(ArrayString):
    """
    功能：获取 每一个 row 的起始符
    参数：NULL
    返回值：每一个 row 的起始符
    """

    def _get_row_start_char(self):
        return ""

    """
    功能：获取 每一个 row 的终止符
    参数：NULL           
    返回值：每一个 row 的终止符
    """

    def _get_row_end_char(self):
        return ""

    """
    功能：获取 每一个 row 的换行符
    参数：NULL           
    返回值：每一个 row 的换行符
    """

    def _get_row_br_char(self):
        return ""

    """
    功能：获取行数据字符串
    参数：
    row：行号
    返回值：某一行的数据的字符串
    """

    def _get_row_string(self, row):
        return "%f" % self.arr[row]


"""
class：D2ArrayString 2维数组转换为字符串
"""


class D2ArrayString(ArrayString):
    """
    功能：获取行数据字符串
    参数：
    row：行号
    返回值：某一行的数据的字符串
    """

    def _get_row_string(self, row):
        # 数组的 height
        height = self.arr.shape[1]

        # 字符串初始化
        row_str = ""

        for col in range(0, height):
            row_str += self._get_col_string(row, col)

            if col < (height - 1):
                row_str += self.split_char

        return row_str

    """
    功能：获取列数据字符串
    参数：
    row：行号
    col：列号
    返回值：某一列的数据的字符串
    """

    def _get_col_string(self, row, col):
        return "%f" % self.arr[row, col]


"""
class：D3ArrayString 3维数组转换为字符串
"""


class D3ArrayString(D2ArrayString):
    """
    功能：获取列数据字符串
    参数：
    row：行号
    col：列号
    返回值：某一列的数据的字符串
    """

    def _get_col_string(self, row, col):
        # 数据的深度
        depth = self.arr.shape[2]

        # 字符串初始化
        col_str = self.start_char

        for kin in range(0, depth):
            col_str += self._get_kin_string(row, col, kin)

            if kin < (depth - 1):
                col_str += self.split_char

        col_str += self.end_char

        return col_str

    """
    功能：获取进数据字符串
    参数：
    row：行号
    col：列号
    kin: 进号
    返回值：某一列的数据的字符串
    """

    def _get_kin_string(self, row, col, kin):
        return "%f" % self.arr[row, col, kin]
