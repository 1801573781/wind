"""
Function：Convolution Neural Network
Author：lzb
Date：2021.01.10
"""

import numpy as np
import operator

from nn.neural_network import NeuralNetwork
from gl import errorcode
from gl.common_enum import ArrayDim
from gl.common_function import rand_array_3

from cnn.convolution import Convolution, ConvolutionType, Reversal, cal_cvl_wh

"""
class：CVLNeuralNetwork，卷积神经网络
说明：
1、继承自 NeuralNetwork
2、重载 _modify_wb 函数

特别说明：
1、每一层的 w，还是一个 matrix，不过它的定义是：卷积核
2、每一层的神经元个数，将由上一层的神经元个数和卷积参数确定（w，s，padding），而不是由外部输入
3、卷积参数：步长 s = 1，补零 padding = 0
4、激活函数是 ReLU 函数
5、输入样本 sx，为了计算方便，将是1个3维矩阵，为了方便理解，可以这么认为：
A. 第1维：图像像素的 x 坐标
B. 第2维：图像像素的 y 坐标
C. 第3维：图像像素的颜色值。如果是 RGB 图像，则第3维有3个值（r, g, b），如果是灰度图像，则第3维只有1个值（gray）
6、从这个意义上讲，参数 sx_dim 将变为1个”3元祖“：sx_dim[0] 表示图像的宽度，sx_dim[1] 表示图像的高度，sx_dim[2] 表示图像的颜色深度
7、输出样本，暂时还定义为分类，所以它还是一个1维向量。因此，sy_dim 还是表示1维向量的元素个数（1维向量的维度）
"""


class CVLNeuralNetwork(NeuralNetwork):
    # 可以理解为图像宽度
    width = 1

    # 可以理解为图像高度
    height = 1

    # 可以理解为颜色深度（颜色维度）
    depth = 1

    # 卷积对象
    cvl = None

    # 每一层神经网络的输出（经过激活函数以后的输出），a 是一个三维数组
    a_list = None

    # 卷积步长
    s = 1

    # 卷积两端补齐长度
    padding = 0

    # 卷积类型
    cvl_type = ConvolutionType.Narrow

    # 是否翻转卷积
    rev = Reversal.NO_REV

    def __init__(self, cvl, activation, last_hop_activation=None, loss=None):
        """
        构造函数
        :param cvl: 卷积对象
        :param activation: 激活函数对象
        :param last_hop_activation: 最后一跳激活函数对象
        :param loss: 损失函数对象
        """

        self.cvl = cvl

        self.activation = activation

        if last_hop_activation is not None:
            self.last_hop_activation = last_hop_activation

        if loss is not None:
            self.loss = loss

    """
    功能：参数校验
    参数：NULL    
    返回值：错误码    
    """

    def _valid(self):
        # 调用父类的 _valid
        err = super()._valid()

        if errorcode.SUCCESS != err:
            return err

        # 校验 w_shape_list

        if self._w_shape_list is None:
            return errorcode.FAILED

        if 0 >= len(self._w_shape_list):
            return errorcode

        # 这里只处理3维数组
        shape = self._w_shape_list[0]

        if 3 != len(shape):
            return errorcode.FAILED

        if (0 >= shape[0]) or (0 >= shape[1]) or (0 >= shape[1]):
            return errorcode.FAILED

        return errorcode.SUCCESS

    """
    功能：校验每层神经元
    参数：NULL    
    返回值：错误码
    说明：对于卷积神经网络来说，这里不需要校验
    """

    def _valid_layer_neuron(self):
        return errorcode.SUCCESS

    """
    功能：校验样本
    参数：NULL    
    返回值：错误码
    说明：理解卷积神经网络，理解卷积对图像的处理，恐怕是从样本的校验开始
    """

    def _valid_sample(self):
        # 1 输入样本的数量与输出样本的数量，须相同（equal with parent class）
        len1 = len(self._sx_list)
        len2 = len(self._sy_list)

        if len1 != len2:
            return errorcode.FAILED

        # 2 样本数量，须 >= 1（same as parent class）
        sample_count = len(self._sx_list)
        if 1 > sample_count:
            return errorcode.FAILED

        # 3. 样本数组维度（different with parent class）

        # 3.1 输入数组维度
        sx_dim = self._sx_list[0].shape

        # 输入样本必须是3维样本：图像的宽度、高度，颜色的深度
        if ArrayDim.THREE.value != len(sx_dim):
            return errorcode.FAILED

        width = sx_dim[0]
        height = sx_dim[1]
        depth = sx_dim[2]

        # 图像宽度/高度须大于0
        if (0 > width) or (0 > height):
            return errorcode.FAILED

        # 颜色深度须介于1~3之间
        if (3 < depth) or (1 > depth):
            return errorcode.FAILED

        # 3.2 每一个输入/输出样本的维度
        for i in range(0, sample_count):
            shape_in = self._sx_list[i].shape
            shape_out = self._sy_list[i].shape

            # 输入样本的维度须相等(都是3维)
            if not operator.eq(sx_dim, shape_in):
                return errorcode.FAILED

            # 输入样本的宽度、高度、深度
            if (shape_in[0] != width) or (shape_in[1] != height) or (shape_in[2] != depth):
                return errorcode.FAILED

            """
            # 输出样本的向量维度
            if shape_out[0] != sy_dim:
                return errorcode.FAILED

            # 输出样本只能有1列（因为是个向量）
            if shape_out[1] != 1:
                return errorcode.FAILED
            """

        return errorcode.SUCCESS

    """
    功能：初始化其它参数
    参数：NULL   
    返回值：错误码
    """

    def _init_other_para(self):
        # 样本数量
        self.sample_count = len(self._sx_list)

        # 神经网络输入，维度(3维)
        self.sx_dim = self._sx_list[0].shape

        # 图像宽度，高度，深度
        self.width = self.sx_dim[0]
        self.height = self.sx_dim[1]
        self.depth = self.sx_dim[2]

        # 神经网络输出，向量维度
        # self.sy_dim = self.neuron_count_list[self.layer_count - 1]

        # 初始化 self.layer_count
        self.layer_count = len(self._w_shape_list)

        # 初始化 W, B
        self._init_w_b()

        return errorcode.SUCCESS

    """
    功能：初始化 W, B
    参数：NULL
    返回值：NULL
    
    特别说明，这里就假设 w 是 3维数组
    """

    def _init_w_b(self):
        """
        # 每一层 w、B 参数，w 是个3维数组，b 是个3维数组
        self.W = list()
        self.B = list()

        # 第1层

        w0 = np.asarray([[[25390.232890], [32600.740020], [10454.593038]],
                        [[30643.149877], [32487.544503], [16065.941808]],
                        [[41348.898949], [36431.949742], [27283.439987]]])

        b0 = np.asarray([[[0.715101], [1.974137], [2.387902], [2.406673], [3.348215], [-10.564372], [-132.415254], [46.503218], [-32.432184], [18.294964], [-36.348997], [0.348342], [2.586671], [2.622596], [2.458631], [3.156726], [1.946428], [0.752287]],
                        [[1.514352], [2.342578], [3.707341], [39.041690], [-38.723104], [-47.723642], [-115.248273], [-99.871424], [-152.845305], [-111.040647], [-70.604176], [32.827495], [24.381645], [-0.001868], [2.902057], [3.329338], [2.302090], [1.010418]],
                        [[0.568814], [2.826865], [5.198503], [-13.100174], [-43.153670], [-154.221700], [-145.123210], [157.504046], [228.561646], [69.061588], [-203.446210], [-275.679652], [61.166458], [-18.368180], [5.161747], [5.657815], [4.351550], [2.184629]],
                        [[0.486020], [3.011486], [5.066088], [4.223942], [-14.907690], [-90.070929], [31.536354], [-146.216915], [-14.987736], [-73.741928], [-57.588058], [35.667701], [195.403711], [-34.780445], [5.565671], [5.299858], [4.590948], [2.326066]],
                        [[0.947281], [2.692973], [5.556650], [-44.771759], [43.432168], [-99.682256], [218.365964], [79.267415], [18.582189], [74.577620], [-56.796083], [-192.641990], [-91.100124], [95.078099], [5.541073], [5.638280], [4.345163], [2.066995]],
                        [[1.080044], [2.372102], [4.724972], [-24.423891], [26.279679], [-42.656457], [33.158186], [-323.842043], [80.057549], [359.937898], [-207.532690], [-132.315522], [-41.769204], [149.593343], [5.570367], [5.354347], [4.654465], [2.907691]],
                        [[0.979589], [3.237929], [4.966936], [-24.423275], [-6.236975], [26.652793], [-138.916726], [4.418455], [281.741575], [-393.734309], [-45.034125], [-37.180427], [1.239852], [34.116887], [5.114519], [5.569703], [4.549612], [1.901177]],
                        [[0.799670], [3.332840], [-44.653020], [3.575846], [-250.615706], [55.692310], [-199.964618], [491.465009], [-447.308779], [-147.283831], [208.762148], [-61.735292], [-15.072089], [32.752241], [4.778338], [5.464183], [4.514829], [2.817789]],
                        [[0.478607], [2.909340], [-24.623380], [-117.138588], [-81.262974], [118.171833], [-69.025740], [-193.504652], [-201.839825], [442.458934], [-125.219486], [-4.906710], [-86.720601], [35.753252], [4.366797], [4.719782], [4.330090], [2.349417]],
                        [[1.164661], [3.281426], [25.286289], [-126.251881], [87.320095], [-160.831114], [116.245030], [-120.666833], [431.987329], [-408.734829], [40.001349], [66.621660], [52.160060], [48.164983], [5.073918], [5.408376], [3.864608], [2.283226]],
                        [[1.297018], [3.335228], [38.581514], [-116.641404], [-90.149115], [-42.054242], [54.744539], [-45.994365], [142.602298], [33.522939], [240.972904], [-199.196715], [84.086153], [-18.958762], [5.217190], [4.481591], [3.918908], [1.997751]],
                        [[0.995451], [2.988009], [128.279363], [-218.173833], [-168.188990], [35.186904], [50.476106], [93.070550], [-64.108372], [143.378322], [-92.980328], [-48.983423], [124.499482], [20.739072], [4.869956], [5.163259], [4.716783], [2.028799]],
                        [[0.526981], [3.297934], [-22.840065], [6.762774], [-185.508998], [250.757581], [-147.651766], [9.101813], [69.327342], [113.361692], [-164.066700], [41.268461], [-35.994979], [20.726219], [5.045523], [5.289321], [4.225684], [2.794068]],
                        [[0.949268], [2.730943], [-16.460908], [65.669214], [-497.128994], [19.735885], [-92.378520], [371.301715], [-193.555743], [-69.007191], [73.601697], [-0.809937], [-62.950916], [19.168995], [4.417340], [5.235519], [4.453848], [2.437773]],
                        [[0.722933], [2.613846], [5.432945], [12.037417], [27.823924], [-72.737513], [15.356950], [-216.919956], [9.106417], [129.196234], [-113.426176], [-72.282017], [-49.113307], [25.712918], [5.098364], [5.335638], [4.257222], [1.963039]],
                        [[0.726214], [2.799415], [4.800972], [-0.467041], [38.506243], [-209.261447], [213.159687], [-30.505563], [119.592213], [-227.553733], [-20.022255], [109.512983], [6.492038], [36.623059], [4.873884], [4.915301], [4.759495], [2.490648]],
                        [[0.156950], [0.662247], [1.735038], [2.830938], [-18.219101], [-13.019722], [80.860724], [-149.027332], [104.635323], [41.798846], [2.488566], [2.518907], [2.339941], [2.243666], [2.585978], [2.863966], [1.740151], [0.833265]],
                        [[-0.241144], [0.728821], [2.350007], [1.644242], [3.851216], [39.976552], [-11.887204], [-83.918601], [48.074015], [2.133532], [1.539688], [1.232162], [0.862038], [1.843876], [2.239465], [1.634374], [1.791269], [0.792765]]])

        self.W.append(w0)
        self.B.append(b0)

        # 第2层

        w1 = np.asarray([[[0.835699], [0.365229], [-0.026167]],
                        [[0.103471], [0.143597], [0.352764]],
                        [[0.573277], [0.772177], [0.268917]]])

        b1 = np.asarray([[[2.136738], [2.139396], [2.145890], [2.138232], [2.135568], [-1.946132], [-2.542707], [-1.946132], [-1.946132], [-1.164288], [2.134440], [2.133318], [2.152019], [2.138849], [2.141795], [2.142692]],
                        [[2.133736], [2.132530], [2.149870], [-1.184384], [-1.956101], [-3.015717], [-2.574270], [-2.587753], [-3.454032], [-3.455468], [-2.587753], [-1.164662], [2.139826], [2.139663], [2.147200], [2.134815]],
                        [[2.142973], [2.139627], [2.137512], [2.134991], [-1.060312], [-2.024394], [-3.062343], [-1.424030], [-2.174933], [-3.033219], [-2.059833], [-2.051925], [2.136854], [2.152195], [2.135529], [2.136433]],
                        [[2.149040], [2.139556], [2.135367], [2.141595], [-1.234372], [-2.602612], [-2.650375], [-1.234372], [-3.454728], [-3.453093], [-1.164662], [-2.051925], [2.134548], [2.139809], [2.141627], [2.133997]],
                        [[2.138855], [2.135000], [2.136424], [-1.234372], [-2.061865], [-2.602612], [-2.141553], [-3.870496], [-3.905406], [-2.629927], [2.149576], [-1.297292], [2.139074], [2.141290], [2.133310], [2.142004]],
                        [[2.144858], [2.142510], [2.140731], [-1.234372], [-2.087731], [-1.164662], [-3.457859], [-4.374313], [-2.525941], [-2.108335], [2.133165], [-1.297292], [2.133461], [2.141146], [2.132676], [2.133266]],
                        [[2.140040], [2.136385], [2.138029], [-1.234372], [-2.087731], [-2.029550], [-3.053633], [-3.031705], [-1.424030], [-1.424030], [-2.051209], [-1.297292], [2.137655], [2.152564], [2.133423], [2.147263]],
                        [[2.132389], [2.152917], [-1.234372], [-1.297292], [-3.055163], [-3.462905], [-3.869296], [-3.451530], [-2.625860], [-2.625860], [-2.597253], [-1.234372], [2.147652], [2.142579], [2.148871], [2.134599]],
                        [[2.141013], [2.145007], [-1.234372], [-2.079792], [-3.037827], [-2.531789], [-2.583552], [-2.602957], [-1.226741], [-2.109051], [-2.582498], [2.142141], [2.136362], [2.153061], [2.150760], [2.144164]],
                        [[2.134615], [2.133026], [2.133945], [-2.051925], [-2.572881], [-1.060312], [-1.345905], [-2.061865], [-2.641931], [-2.649293], [-2.582498], [-1.164662], [2.140825], [2.137089], [2.133941], [2.140801]],
                        [[2.141249], [2.138881], [-1.164288], [-2.587753], [-2.546910], [2.133301], [-2.100488], [-1.234372], [-2.641931], [-1.297292], [-1.164288], [-1.164662], [2.141555], [2.148685], [2.142920], [2.140487]],
                        [[2.134918], [2.136673], [-2.032346], [-3.455468], [-1.976131], [-1.226741], [-2.100488], [-3.069533], [-3.069533], [2.135746], [-1.164288], [-1.164662], [2.141014], [2.147591], [2.137579], [2.144606]],
                        [[2.138931], [2.139029], [-1.253613], [-3.895517], [-2.582498], [-3.051963], [-3.059014], [-3.069533], [-1.164288], [-1.993053], [-1.993053], [-1.164662], [2.142583], [2.144081], [2.141265], [2.135053]],
                        [[2.145523], [2.134445], [2.145868], [-1.988794], [-3.029480], [-3.875695], [-3.451140], [-2.587534], [-2.525941], [-1.946833], [-1.060312], [-1.060312], [2.152394], [2.150089], [2.135066], [2.144580]],
                        [[2.144693], [2.142972], [2.134324], [2.141563], [2.139191], [-1.345905], [-2.024394], [-1.164662], [2.135452], [2.134267], [2.152031], [2.147196], [2.150251], [2.133119], [2.136882], [2.145675]],
                        [[2.142513], [2.132361], [2.151419], [2.144383], [-1.345905], [-1.345905], [-1.234372], [2.137358], [2.133374], [2.145218], [2.145559], [2.151259], [2.133092], [2.135684], [2.143144], [2.136400]]])

        self.W.append(w1)
        self.B.append(b1)

        """

        # 每一层 w、B 参数，w 是个3维数组，b 是个3维数组
        self.W = list()
        self.B = list()

        # 针对每一层进行初始化
        b = 0
        for layer in range(0, self.layer_count):
            # 2.1 每一层的卷积核
            width = self._w_shape_list[layer][0]
            height = self._w_shape_list[layer][1]

            # 如果是第一层，depth = 输入层的 depth
            if 0 == layer:
                depth = self._w_shape_list[layer][2]
            # 否则的话，depth = 1
            else:
                depth = 1

            w = rand_array_3(width, height, depth)
            # w = np.zeros([width, height, depth])
            self.W.append(w)

            # 2.2 每一层的 b

            # 如果是第一层，x 就是样本输入
            if 0 == layer:
                x = self._sx_list[0]
            # 否则的话，x 是上一层的输出
            # 上一层的输出的 width，height 等同于 b
            else:
                x = b

            width, height = cal_cvl_wh(w, x, self.s)

            # 每一层的b，都是 [width, height, depth] 3维数组
            depth = 1  # b 的 depth = 1
            b = rand_array_3(width, height, depth)
            # b = np.zeros([width, height, depth])

            self.B.append(b)

    """
    功能：计算某一层神经网络的输出
    参数：
    x：该层神经网络的输入，x 是一个3维数组
    w: 该层神经网络的 w 参数, w 是一个3维数组
    b：该层神经网络的 b 参数，b 是一个3维数组
    返回值：y，该层神经网络的输出（sigmoid(cvl(w, x) + b)）， y 是一个3维数组
    """

    def _calc_layer(self, x, layer):
        # 1、获取该层的参数：w, b
        w = self.W[layer]
        b = self.B[layer]

        # 2、计算卷积结果
        y, err = self.cvl.convolution_sum_depth(w, x)
        # y, err = cvl.convolution(w, x)

        # 3. y = y + b
        y_width = y.shape[0]
        y_height = y.shape[1]
        y_depth = y.shape[2]

        for i in range(0, y_width):
            for j in range(0, y_height):
                for k in range(0, y_depth):
                    y[i, j, k] += b[i, j, k]

        # 针对每一个元素，调用激活函数
        for i in range(0, y_width):
            for j in range(0, y_height):
                for k in range(0, y_depth):
                    y[i, j, k] = self.activation.active(y[i, j, k])

        return y

    """
    功能：修正 W，B
    参数：
    nn_y_list：神经网路计算的每一层结果，nn_y 是一个向量
    sx：训练样本的输入，sx 是一个向量
    sy：训练样本的输出，sy 是一个向量 
    返回值：NULL
    """

    def _modify_wb(self, nn_y_list, sx, sy):
        # 1. 后向传播，计算 ksi_list
        ksi_list = self.__bp(nn_y_list, sy)

        # 2. 通过 ksi_list，修正 W，B
        self.__modify_wb_by_ksi_list(ksi_list, sx, nn_y_list)

    """
    功能：后向传播，计算 ksi_list
    参数：
    nn_y_list：神经网路计算的每一层结果，nn_y 是一个3维数组    
    sy：训练样本的输出，sy 是一个3维数组
    返回值：ksi_list
    说明：
    1、ksi(代表希腊字母，音：科赛)，是一个3维数组，每层都有，代表目标函数 E 对每一层中间输出的偏导
    2、ksi_list 记录每一层的 ksi
    """

    def __bp(self, nn_y_list, sy):
        # 1. 初始化 ksi_list
        ksi_list = [0] * self.layer_count

        # 2. 计算最后一层 ksi

        # 2.1 计算误差(err)：最后一层的计算结果与样本输出结果的比较（计算结果 - 训练样本的输出）
        nn_y_last = nn_y_list[self.layer_count - 1]
        # dy = np.subtract(nn_y_last, sy)  # 不知道3维数组是否可以这样相减
        dy = self.loss.derivative(nn_y_last, sy)

        # 2.2 计算最后一层 ksi

        # 最后一层 ksi：ksi_last，ksi_last 是个[width, height, 1] 3维数组
        width = nn_y_last.shape[0]
        height = nn_y_last.shape[1]
        depth = nn_y_last.shape[2]  # 实际的值，depth = 1

        ksi_last = np.zeros([width, height, depth])

        # 计算 ksi_last 每一个元素
        for k in range(0, depth):
            for i in range(0, width):
                for j in range(0, height):
                    ksi_last[i, j, k] = dy[i, j, k] * self.activation.derivative(nn_y_last[i, j, k])

        # 将 ksi_last 放置入 ksi_list
        ksi_list[self.layer_count - 1] = ksi_last

        # 3. 反向传播，计算：倒数第2层 ~ 第1层的 ksi
        for layer in range(self.layer_count - 2, -1, -1):
            # 下一层的 ksi
            ksi_next = ksi_list[layer + 1]

            # 下一层的 w
            w = self.W[layer + 1]

            # 当前层的 ksi
            ksi_cur, dy = self.cvl.convolution(w, ksi_next, Reversal.REV, ConvolutionType.Wide)

            # 将当前层计算出的 ksi 放置到 ksiList
            ksi_list[layer] = ksi_cur

        # return 计算结果
        return ksi_list

    ''''''

    def __modify_wb_by_ksi_list(self, ksi_list, sx, nn_y_list):
        """
        功能：修正 W，B
        参数：
        ksi_list：每一层的 ksi 的列表，ksi 是一个3维数组
        sx：输入样本，sx 是一个3维数组
        nn_y_list：神经网络的每一层的计算结果列表，nn_y 是一个3维数组
        返回值：NULL
        """

        # 逐层修正
        for layer in range(0, self.layer_count):
            # 当前层 w, b, ksi
            w = self.W[layer]
            b = self.B[layer]
            ksi = ksi_list[layer]

            # 上一层的输入
            if 0 == layer:
                v = sx
            else:
                v = nn_y_list[layer - 1]

            # 损失函数针对当前层的 w 的偏导(partial derivative)，w_pd 是1个3维数组
            w_pd, err = self.cvl.convolution(ksi, v)

            # 修正当前层的 w
            self.W[layer] = np.subtract(w, self._rate * w_pd)  # 不知道3维数组是否可以这样相减

            # 损失函数针对当前层的 b 的偏导(partial derivative)，b_pd 等于 ksi
            b_pd = ksi

            # 修正当前层的 b
            self.B[layer] = np.subtract(b, self._rate * b_pd)  # 不知道3维数组是否可以这样相减
