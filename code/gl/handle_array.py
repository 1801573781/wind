"""
Function：处理数组
Author：lzb
Date：2021.01.30
"""


def handle_arr(arr, f, *args):
    """
    功能：递归处理数组\n
    参数：\n
    arr：待处理的数组\n
    f：处理函数\n
    *args：变参
    返回值：NULL\n
    """

    # 数组的维度
    shape = arr.shape
    dim = len(shape)

    # 第1维元素的个数
    count = shape[0]

    # 如果是1维，则开始处理
    if 1 == dim:
        for i in range(0, count):
            # 将 arr[i] 放在 args 的第一个位置
            args2 = (arr[i],) + args
            arr[i] = f(args2)
    # 如果不是1维，则需要递归：
    else:
        for i in range(0, count):
            a = arr[i]

            handle_arr(a, f, *args)


def sum_arr(arr, s, f=None, *args):
    """
    功能：递归求和数组\n
    参数：\n
    arr：待处理的数组\n
    s: sum 结果的保存。s 是一个一行一列的数组，为了引用传递\n
    f：处理函数\n
    *args：变参\n
    返回值：NULL\n
    """

    # 数组的维度
    shape = arr.shape
    dim = len(shape)

    # 第1维元素的个数
    count = shape[0]

    # 如果是1维，则开始处理
    if 1 == dim:
        for i in range(0, count):
            # 如果 f 为空，则直接求和
            if f is None:
                s[0] += arr[i]
            # 否则，先处理以后，再求和
            else:
                args2 = (arr[i],) + args
                s[0] += f(args2)
    # 如果不是1维，则需要递归：
    else:
        for i in range(0, count):
            a = arr[i]

            sum_arr(a, s, f, args)


''''''


def get_arr_item(arr, index):
    """
    功能：求数组的某个索引的值
    参数：
    arr：n维数组
    index：n维数组的索引,1维数组，比如 [2, 3, 4]，表示 arr[2][3][4]
    返回值：数组的某个索引的值， arr[index]
    """

    # 索引所包含元素的个数
    count = len(index)

    # 用递归方法，获取 arr[index]
    a = arr[index[0]]
    for i in range(1, count):
        a = a[index[i]]

    return a
