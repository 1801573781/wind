"""
Function：卷积神经测试
Author：lzb
Date：2021.01.14

特别说明：来到了卷积网络，这几天编码的效率有点低
"""


from cnn.convolution import Convolution, Reversal, ConvolutionType
from cnn.cvl_nn import CVLFNN

from gl.common_enum import ArrayDim
from gl.common_function import *
from my_image import my_image
from my_image.my_image import show_file, gray_file, show_data, ImageDataType, get_data
from activation.normal_activation import Sigmoid


from activation.last_hop_activation import LastHopActivation
from loss.loss import MSELoss

from tkinter import messagebox

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
功能：测试卷积神经网络
参数：NULL 
返回值：NULL
"""


def test_cvl_nn():
    # 1. 构建训练输入样本/输出样本

    sx_list = list()  # 输入样本列表
    sy_list = list()  # 输出样本列表

    for i in range(0, 10):
        # 1.1 输入样本

        # 图像文件名
        file_name = "../picture/number/"
        file_name += "%d_sx.bmp" % i

        # 取灰度值
        data, err = gray_file(file_name, ArrayDim.THREE)

        # 将图像数据中的0转换为极小值
        my_image.array_0_tiny(data)

        # 显示灰度图像
        gray = my_image.array_3_2(data)
        # show_data(gray, ImageDataType.GRAY)

        # 归一化
        sx = my_image.normalize(data, my_image.NormalizationType.NORMAL)

        # 显示归一化灰度图像
        gray = my_image.array_3_2(sx)
        # show_data(gray, ImageDataType.GRAY)

        # 加入训练输入样本列表
        sx_list.append(sx)

        # 1.2 输出样本

        # 图像文件名
        file_name = "../picture/number/"
        file_name += "%d_sy_2.bmp" % i

        # 取灰度值
        data, err = gray_file(file_name, ArrayDim.THREE)

        # 将图像数据中的0转换为极小值
        my_image.array_0_tiny(data)

        # 显示灰度图像
        gray = my_image.array_3_2(data)
        # show_data(gray, ImageDataType.GRAY)

        # 归一化
        sy = my_image.normalize(data, my_image.NormalizationType.NORMAL)

        # 显示归一化灰度图像
        gray = my_image.array_3_2(sy)
        # show_data(gray, ImageDataType.GRAY)

        # 加入训练输出样本列表
        sy_list.append(sy)

    # 2. 卷积神经网络的基本参数

    # 每一层网络的神经元个数(这个参数，对于卷积网络而言，没意义)
    neuron_count_list = None

    # 最大循环训练次数
    loop_max = 1

    # 学习效率
    rate = 0.01

    # 卷积核数组大小
    w_shape_list = list()

    w1_shape = [3, 3, 1]
    w2_shape = [3, 3, 1]

    w_shape_list.append(w1_shape)
    w_shape_list.append(w2_shape)

    # 激活函数
    activation = Sigmoid()
    # activation = ReLU()

    # 最后一跳激活函数
    last_hop_activation = LastHopActivation()

    # 损失函数
    loss = MSELoss()

    # 构建卷积对象
    cvl = Convolution()

    # 构建卷积神经网络对象
    cnn = CVLFNN(cvl, activation, last_hop_activation, loss)

    # 3. 训练
    cnn.train(sx_list, sy_list, loop_max, neuron_count_list, rate, w_shape_list)

    # 4. 预测
    py_list = cnn.predict(sx_list, sy_list)

    # 将预测结果显示如初
    count = len(py_list)

    for i in range(0, count):
        py = py_list[i]

        py = my_image.normalize(py, my_image.NormalizationType.REV_NORMAL)

        py = my_image.array_3_2(py)

        show_data(py, ImageDataType.GRAY)


"""
功能：测试卷积神经网络
参数：NULL 
返回值：NULL
"""


def test_cvl_nn_2():
    # 1. 构建训练输入样本/输出样本

    train_sx_list = list()  # 训练样本列表（输入）
    train_sy_list = list()  # 训练样本列表（输出）

    test_sx_list = list()  # 测试样本列表（输入）
    test_sy_list = list()  # 测试样本列表（输入）

    max_count = 1000

    image_path = "../picture/number_cnn/"

    for count in range(0, max_count):
        for number in range(0, 2):
            # 1.1 输入样本

            # 图像路径
            path = image_path + "/" + str(number)

            # 文件名
            sx_file_name = path + "/" + "sx_" + str(number) + "_" + str(count) + ".bmp"

            # 取灰度值
            data, err = gray_file(sx_file_name, ArrayDim.THREE)

            # 将图像数据中的0转换为极小值
            my_image.array_0_tiny(data)

            # 显示灰度图像
            # gray = my_image.array_3_2(data)
            # show_data(gray, ImageDataType.GRAY)

            # 归一化
            sx = my_image.normalize(data, my_image.NormalizationType.NORMAL)

            sx = sx / 40

            # 显示归一化灰度图像
            # gray = my_image.array_3_2(sx)
            # show_data(gray, ImageDataType.GRAY)

            # 加入训练样本列表（输入）
            if count < (max_count - 1):
                train_sx_list.append(sx)
            # 加入测试样本列表（输入）
            else:
                test_sx_list.append(sx)

            # 1.2 输出样本

            # sy 文件名
            sy_file_name = path + "/" + "sy_" + str(number) + ".bmp"

            # 取灰度值
            data, err = gray_file(sy_file_name, ArrayDim.THREE)

            # 将图像数据中的0转换为极小值
            my_image.array_0_tiny(data)

            # 显示灰度图像
            # gray = my_image.array_3_2(data)
            # show_data(gray, ImageDataType.GRAY)

            # 归一化
            sy = my_image.normalize(data, my_image.NormalizationType.NORMAL)

            # 显示归一化灰度图像
            # gray = my_image.array_3_2(sy)
            # show_data(gray, ImageDataType.GRAY)

            # 加入训练样本列表（输出）
            if count < (max_count - 1):
                train_sy_list.append(sy)
            # 加入训练样本列表（输出）
            else:
                test_sy_list.append(sy)

    # 2. 卷积神经网络的基本参数

    # 每一层网络的神经元个数(这个参数，对于卷积网络而言，没意义)
    neuron_count_list = None

    # 最大循环训练次数
    loop_max = 1

    # 学习效率
    rate = 0.01

    # 卷积核数组大小
    w_shape_list = list()

    w1_shape = [3, 3, 1]
    w2_shape = [3, 3, 1]

    w_shape_list.append(w1_shape)
    w_shape_list.append(w2_shape)

    # 激活函数
    activation = Sigmoid()
    # activation = ReLU()

    # 最后一跳激活函数
    last_hop_activation = LastHopActivation()

    # 损失函数
    loss = MSELoss()

    # 构建卷积对象
    cvl = Convolution()

    # 构建卷积神经网络对象
    cnn = CVLFNN(cvl, activation, last_hop_activation, loss)

    # 3. 训练
    # train_sx_group = get_sample_group(train_sx_list)
    # train_sy_group = get_sample_group(train_sy_list)
    # cnn.train(train_sx_group, train_sy_group, loop_max, neuron_count_list, rate, w_shape_list)
    cnn.train(train_sx_list, train_sy_list, loop_max, neuron_count_list, rate, w_shape_list)

    # 4. 预测
    py_list = cnn.predict(test_sx_list, test_sy_list)

    # 将预测结果显示如初
    count = len(py_list)

    for number in range(0, count):
        py = py_list[number]

        py = my_image.normalize(py, my_image.NormalizationType.REV_NORMAL)

        py = my_image.array_3_2(py)

        show_data(py, ImageDataType.GRAY)


''''''


def get_sample_group(sample_list):
    """
    将训练样本，分组。默认是1个训练样本，1组
    :param sample_list:
    :return: 训练样本分组
    """

    sample_group = list()

    count = len(sample_list)

    for i in range(0, count):
        sample = sample_list[i]
        # 1个样本组成1个list
        sl = [sample]
        # 1样本组成的list，是1个分组
        sample_group.append(sl)

    return sample_group


"""
功能：测试卷积神经网络，不经过训练，直接赋值参数
参数：NULL 
返回值：NULL
"""


def test_cvl_nn_without_train():
    # 1. 构建训练输入样本

    # 图像文件名
    file_name = "../picture/number/1_sx.bmp"

    # 取灰度值
    data, err = gray_file(file_name, ArrayDim.THREE)

    # 将图像数据中的0转换为极小值
    my_image.array_0_tiny(data)

    # 显示灰度图像
    gray = my_image.array_3_2(data)
    show_data(gray, ImageDataType.GRAY)

    # 归一化
    sx = my_image.normalize(data, my_image.NormalizationType.NORMAL)

    # 显示归一化灰度图像
    gray = my_image.array_3_2(sx)
    show_data(gray, ImageDataType.GRAY)

    # 训练输入样本
    sx_list = list()
    sx_list.append(sx)

    # 2. 构建训练输出样本

    # 图像数据
    file_name = "../picture/number/8_sy_2.bmp"

    # 取灰度值
    data, err = gray_file(file_name, ArrayDim.THREE)

    # 将图像数据中的0转换为极小值
    my_image.array_0_tiny(data)

    # 显示灰度图像
    gray = my_image.array_3_2(data)
    # show_data(gray, ImageDataType.GRAY)

    # 归一化
    sy = my_image.normalize(data, my_image.NormalizationType.NORMAL)

    # 显示归一化灰度图像
    gray = my_image.array_3_2(sy)
    show_data(gray, ImageDataType.GRAY)

    # 训练输出样本
    sy_list = list()
    sy_list.append(sy)

    # 3. 卷积神经网络的基本参数

    ''' W, B 参数 begin '''

    # 每一层 w、B 参数，w 是个 matrix，b 是个 vector（数据类型也是一个 matrix）
    W = list()
    B = list()

    # 第1层

    w0 = np.asarray([[[50.113265], [93.826943], [115.373675]],
                     [[59.189640], [119.609671], [104.736651]],
                     [[42.784839], [90.116834], [67.911661]]])

    b0 = np.asarray([[[0.394134], [0.305481], [-0.111563], [0.464680], [-0.044974], [-0.249474], [0.344747],
                      [-0.144516], [0.120547], [-0.120140], [0.096371], [0.073717], [0.044348], [-0.385316], [0.416685],
                      [0.548801], [0.290702], [-0.170383]],
                     [[-0.183913], [0.881344], [0.379849], [0.967322], [0.231840], [0.334352], [0.516944], [0.570409],
                      [0.156349], [0.909234], [0.174547], [0.813966], [0.835244], [0.761143], [1.020271], [0.431555],
                      [0.451456], [-0.165681]],
                     [[0.538315], [1.158434], [0.827443], [1.209531], [0.873020], [1.490418], [-0.321991], [-0.546698],
                      [0.387941], [0.714195], [1.914184], [1.262890], [1.031726], [0.665484], [1.111183], [0.780210],
                      [0.800418], [0.016235]],
                     [[0.598855], [1.493059], [1.128118], [1.443648], [0.734596], [1.172370], [-0.883454], [-4.379097],
                      [-3.697614], [-5.167245], [-3.755237], [1.490307], [0.865462], [1.644435], [1.055617], [1.160021],
                      [0.387357], [0.157360]],
                     [[0.198682], [0.914774], [1.392495], [0.880891], [1.134312], [0.968554], [-4.343983], [-7.116327],
                      [-3.834269], [-5.174044], [-4.161452], [0.151857], [0.922105], [1.161507], [0.595232], [0.990366],
                      [1.075088], [-0.258814]],
                     [[0.477665], [1.602952], [1.181507], [0.665496], [1.279437], [1.439306], [-3.930717], [-2.732151],
                      [0.424828], [-3.961826], [-3.768221], [-0.472824], [1.042067], [0.807338], [1.711559], [0.983939],
                      [0.288783], [-0.418911]],
                     [[0.766438], [1.244276], [1.286307], [0.790951], [0.995913], [0.872906], [-3.827127], [-3.361843],
                      [-1.077353], [-4.582801], [-2.303175], [0.031738], [1.287701], [1.044528], [1.383836], [1.283100],
                      [0.241565], [-0.197527]],
                     [[0.411086], [1.588132], [1.143584], [1.515480], [0.525359], [1.285077], [-4.642499], [-4.682207],
                      [-2.799712], [-7.874153], [-3.598426], [0.163200], [1.050818], [1.072107], [1.647231], [1.297448],
                      [0.472967], [-0.365622]],
                     [[1.140368], [1.331310], [1.039651], [1.444111], [0.588199], [1.074860], [-2.961898], [-3.365907],
                      [-9.275475], [-8.768490], [-5.709736], [0.552231], [0.806262], [0.641553], [0.859427], [1.390415],
                      [-0.017225], [0.395236]],
                     [[1.151735], [0.551786], [1.383289], [0.987141], [1.303332], [0.891473], [-2.649088], [-2.601597],
                      [-4.444745], [-6.328936], [-3.561121], [0.097629], [0.956537], [1.460169], [1.674805], [1.392249],
                      [0.491228], [-0.388936]],
                     [[0.351013], [1.419691], [1.283733], [1.161723], [1.707191], [0.643386], [1.431694], [0.691841],
                      [-1.213639], [-7.302778], [-3.354404], [-0.886146], [0.715025], [0.994340], [0.875995],
                      [1.663526], [0.948911], [0.567744]],
                     [[0.857319], [1.086111], [0.814578], [0.970882], [1.697328], [1.316010], [1.051087], [0.414327],
                      [-5.256378], [-5.877422], [-0.179250], [0.544970], [1.264428], [0.604863], [0.999309], [1.183568],
                      [0.366828], [0.408546]],
                     [[0.746337], [1.342136], [1.351453], [1.447988], [1.473613], [1.172948], [1.625175], [-1.097880],
                      [-7.496154], [-3.486610], [-0.854779], [0.951540], [1.028474], [0.571090], [0.728899], [1.214444],
                      [0.993936], [0.392067]],
                     [[0.834584], [0.852879], [0.819633], [1.100310], [1.119684], [0.838549], [0.308638], [-4.381538],
                      [-5.573656], [-0.240430], [0.531746], [1.503448], [0.747390], [1.253126], [0.905447], [1.262200],
                      [0.442319], [0.442514]],
                     [[0.743176], [0.749403], [1.527007], [1.030668], [1.659595], [0.818656], [0.471315], [-6.560603],
                      [-4.153131], [-0.389991], [0.951427], [1.292622], [1.227982], [0.787592], [1.083349], [1.171198],
                      [0.320301], [0.145997]],
                     [[0.336351], [0.971745], [0.870700], [1.566745], [0.849081], [1.463097], [-2.652546], [-2.183096],
                      [-0.495741], [-0.188252], [0.936387], [0.429778], [1.737350], [1.646101], [0.694978], [1.069849],
                      [0.892659], [-0.138695]],
                     [[0.193120], [0.772601], [0.665977], [0.959221], [1.782215], [1.507661], [1.003219], [1.342632],
                      [0.805974], [1.522182], [0.674850], [0.820031], [0.603524], [0.752671], [0.938834], [1.467478],
                      [0.745415], [0.440291]],
                     [[0.452432], [0.504461], [0.544565], [0.432389], [0.740133], [0.067003], [0.893957], [0.292302],
                      [0.586535], [0.411866], [0.240025], [0.279950], [0.342256], [0.439238], [0.082237], [0.079462],
                      [0.350406], [-0.376255]]])

    W.append(w0)
    B.append(b0)

    # 第2层

    w1 = np.asarray([[[0.496053], [0.142468], [0.692607]],
                     [[-0.152569], [0.746855], [0.665359]],
                     [[-0.078141], [0.187796], [0.004870]]])

    b1 = np.asarray([[[0.738316], [0.558320], [0.616535], [0.563276], [0.397948], [0.403014], [0.623214], [0.648124],
                      [0.499537], [0.477057], [0.458188], [0.620421], [0.698390], [0.680706], [0.483114], [0.579076]],
                     [[0.596898], [0.404650], [0.519849], [0.451572], [0.522308], [0.449322], [0.525448], [0.713798],
                      [0.439633], [0.500006], [0.759672], [0.725525], [0.695965], [0.449514], [0.472447], [0.482284]],
                     [[0.523954], [0.755020], [0.640807], [0.670937], [0.458754], [0.624749], [-4.708748], [-4.716255],
                      [-4.719785], [-4.739897], [0.416815], [0.563930], [0.471036], [0.385172], [0.570061], [0.460100]],
                     [[0.488801], [0.445694], [0.726974], [0.642160], [0.389185], [0.542930], [-4.720975], [0.494188],
                      [0.493539], [-4.720184], [0.369179], [0.534885], [0.693592], [0.757328], [0.404759], [0.510594]],
                     [[0.387318], [0.776344], [0.697167], [0.591275], [0.678695], [0.379232], [-4.735606], [0.372957],
                      [0.629077], [-4.710119], [0.417072], [0.458907], [0.603948], [0.368090], [0.548514], [0.777818]],
                     [[0.386241], [0.504484], [0.492332], [0.515327], [0.734786], [0.410316], [-4.724017], [0.653110],
                      [0.631986], [-4.728177], [0.658651], [0.705176], [0.620827], [0.670017], [0.385353], [0.462904]],
                     [[0.406522], [0.407609], [0.434104], [0.548003], [0.671716], [0.470964], [-4.705063], [0.561526],
                      [-4.714012], [-4.730049], [0.713027], [0.415023], [0.527258], [0.653842], [0.641143], [0.730861]],
                     [[0.399245], [0.624423], [0.586776], [0.640310], [0.531442], [0.522265], [-4.731969], [-4.713455],
                      [-4.737818], [-4.726882], [0.520856], [0.613984], [0.503019], [0.548040], [0.382262], [0.712375]],
                     [[0.604145], [0.507184], [0.657208], [0.454861], [0.526420], [0.715608], [0.425322], [0.525826],
                      [0.519881], [-4.711324], [0.763128], [0.420057], [0.740006], [0.488205], [0.443485], [0.460116]],
                     [[0.546870], [0.772369], [0.414549], [0.415155], [0.480857], [0.598054], [0.432484], [0.766837],
                      [-4.708531], [-4.714308], [0.559024], [0.703133], [0.608846], [0.459609], [0.689914], [0.389198]],
                     [[0.422926], [0.502196], [0.720415], [0.539214], [0.635489], [0.552064], [0.608547], [0.535145],
                      [-4.723283], [0.467894], [0.476116], [0.527392], [0.498777], [0.663084], [0.404508], [0.373993]],
                     [[0.425400], [0.474075], [0.628678], [0.415942], [0.379077], [0.401436], [0.551009], [-4.741356],
                      [-4.714145], [0.649679], [0.422052], [0.414361], [0.758212], [0.771625], [0.563505], [0.599782]],
                     [[0.534033], [0.497468], [0.427890], [0.373989], [0.604646], [0.600752], [0.368556], [-4.724289],
                      [0.513206], [0.409971], [0.700321], [0.451720], [0.419490], [0.556964], [0.546020], [0.482902]],
                     [[0.392839], [0.550031], [0.598946], [0.515958], [0.419458], [0.400969], [-4.713971], [-4.738337],
                      [0.695640], [0.406998], [0.538457], [0.599610], [0.416045], [0.520507], [0.599289], [0.683523]],
                     [[0.586383], [0.436032], [0.651763], [0.542701], [0.438006], [0.594131], [0.400705], [0.745671],
                      [0.656867], [0.559175], [0.689486], [0.520835], [0.385378], [0.659122], [0.651156], [0.470575]],
                     [[0.405100], [0.716354], [0.549392], [0.384638], [0.413940], [0.525830], [0.380278], [0.649017],
                      [0.428349], [0.389288], [0.761551], [0.657696], [0.416118], [0.655166], [0.488445], [0.488434]]])

    W.append(w1)
    B.append(b1)

    ''' W, B 参数 end '''

    # 激活函数
    activation = Sigmoid()
    # activation = ReLU()

    # 最后一跳激活函数
    last_hop_activation = LastHopActivation()

    # 损失函数
    loss = MSELoss()

    # 构建卷积对象
    cvl = Convolution()

    # 构建卷积神经网络对象
    cnn = CVLFNN(cvl, activation, last_hop_activation, loss)

    # 参数赋值
    cnn.stub_set_para(0, None, W, B, activation)

    # 预测
    py_list = cnn.predict(sx_list, sy_list)

    # 将预测结果显示如初
    py = py_list[0]

    py = my_image.normalize(py, my_image.NormalizationType.REV_NORMAL)

    py = my_image.array_3_2(py)

    messagebox.showinfo("提示", "预测结果")

    show_data(py, ImageDataType.GRAY)
