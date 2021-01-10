"""
Function：softmax 函数
Author：lzb
Date：2021.01.10
"""

"""
class：SoftMax
"""


class SoftMax:
    # x list
    x_list = None

    # x list 中各个 x 相加
    sum = 0

    """
    功能：构造函数
    参数：
    x_list：待计算的各个 x 的值     
    返回值：NULL    
    """

    def __init__(self, x_list):
        # 1. 成员变量赋值
        self.x_list = x_list

        # 2. 校验
        self._valid()

        # 3. 计算各个 x 之和
        self.sum = self._sum()

    """
    功能：校验
    参数：NULL        
    返回值：NULL
    说明：如果校验不合法，则抛异常    
    """

    def _valid(self):
        # x_list 不能为空
        if self.x_list is None:
            raise Exception("x_list is none")

        # x_list 长度不能为0
        if 0 == len(self.x_list):
            raise Exception("x_list len = 0")

    """
    功能：计算各个 x 之和
    参数：NULL        
    返回值：各个 x 之和        
    """

    def _sum(self):
        size = len(self.x_list)

        s = 0
        for i in range(0, size):
            # 首先判断各个元素，每个元素都不能小于0
            if 0 > self.x_list[i]:
                raise Exception("x_list[i] < 0", i)

            # 然后再相加
            s += self.x_list[i]

        # 最后判断所求之和 s，也不能等于0
        if s < 1e-6: # float 通过这个方法，近似判断其是否为0
            raise Exception("sum = 0")

        return s

    """
    功能：计算第 i 个元素的概率
    参数：NULL        
    返回值： 第 i 个元素的概率
    """

    def probability(self, i):
        # 首先判断 i 的合法性
        if (0 > i) or (i >= len(self.x_list)):
            raise Exception("invalid index", i)

        # 计算概率
        p = self.x_list[i] / self.sum

        return p
