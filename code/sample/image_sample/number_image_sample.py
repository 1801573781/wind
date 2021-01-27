"""
Function：生成数字图像文件样本（0~9）
Author：lzb
Date：2021.01.27
"""

from PIL import Image
import random

from PIL import ImageDraw
from PIL import ImageFont

"""
class：NumberImageSample，生成数字图像文件样本（0~9）
说明：
1、暂时只生成 0~9 图像
2、图像是 RGB，字体颜色是黑色，底色是灰度
3、生成图像的目录，暂时写死
"""


class NumberImageSample:
    # sx image width
    sx_width = 0

    # sx image height
    sx_height = 0

    # sy image width
    sy_width = 0

    # sy image height
    sy_height = 0

    # image path
    image_path = "./../../picture/number2"

    # 图像格式
    image_format = "bmp"

    # min number(最小的数字，0)
    min_number = 0

    # max number(最大的数字，9)
    max_number = 9

    # 每个数字生成的 sx 样本数量
    sx_count_per_number = 1000

    # 黑色
    black = 0

    # 白色
    white = 255

    # 字体
    font = "Arial.ttf"

    # 字体大小
    font_size = 10

    """
    功能：构造函数
    参数：TODO 补充
    返回值：NULL
    """

    def __init__(self, sx_width, sx_height, sy_width, sy_height):
        self.sx_width = sx_width
        self.sx_height = sx_height
        self.sy_width = sy_width
        self.sy_height = sy_height

    """
    功能：生成数字图像
    参数：TODO 补充
    返回值：NULL
    """

    def create_image(self, image_path=None, sx_count_per_number=1000):
        # 1. 参数赋值
        if image_path is not None:
            self.image_path = image_path

        self.sx_count_per_number = sx_count_per_number

        # 2. 生成图像
        for i in range(self.min_number, self.max_number + 1):
            # 2.1 生成 sx init 图像
            self._create_sx_init_image(i)

            # 2.2 生成 sx 图像
            for j in range(0, self.sx_count_per_number):
                self._create_sx_image(i, j)

            # 2.3 生成 sy 图像
            self._create_sy_image(i)

    """
    功能：生成数字 sx 原始图像
    参数：TODO 补充
    返回值：NULL
    """

    def _create_sx_init_image(self, number):
        # 图像路径
        path = self.image_path + "/" + str(number)

        # 文件名
        file_name = path + "/" + "sx_" + str(number) + "_init" + ".bmp"

        # image width
        width = self.sx_width

        # image height
        height = self.sx_height

        # 是否有噪声(无噪声)
        noise = False

        # 创建图像文件
        self._create_image(file_name, width, height, number, noise)

    """
    功能：生成数字 sx 图像
    参数：TODO 补充
    返回值：NULL
    """

    def _create_sx_image(self, number, count):
        # 图像路径
        path = self.image_path + "/" + str(number)

        # 文件名
        file_name = path + "/" + "sx_" + str(number) + "_" + str(count) + ".bmp"

        # image width
        width = self.sx_width

        # image height
        height = self.sx_height

        # 是否有噪声(无噪声)
        noise = True

        # 创建图像文件
        self._create_image(file_name, width, height, number, noise)

    """
    功能：生成数字 sy 图像
    参数：TODO 补充
    返回值：NULL
    """

    def _create_sy_image(self, number):
        # 图像路径
        path = self.image_path + "/" + str(number)

        # sx 文件名
        sx_file_name = path + "/" + "sx_" + str(number) + "_init" + ".bmp"

        # sy 文件名
        sy_file_name = path + "/" + "sy_" + str(number) + ".bmp"

        # sx image
        sx_img = Image.open(sx_file_name)

        # sy image(裁剪 sx image)
        left = (self.sx_width - self.sy_width) / 2
        top = (self.sx_height - self.sy_height) / 2
        right = left + self.sy_width
        bottom = top + self.sy_height

        sy_img = sx_img.crop([left, top, right, bottom])

        # 保存
        sy_img.save(sy_file_name, format=self.image_format)

        sx_img.close()

    """
    功能：生成数字 sx 原始图像
    参数：TODO 补充
    返回值：NULL
    """

    def _create_image(self, file_name, width, height, number, noise):
        # 1. 新建一个图像(RGB, 白底)
        image = Image.new('RGB', (width, height), self._white_color())

        # 2. 创建 draw 对象:
        draw = ImageDraw.Draw(image)

        # 3. 创建 font 对象:
        # font = ImageFont.truetype(self.font, self.font_size)

        # 5. 填充每个像素:
        if noise:
            for x in range(0, width):
                for y in range(0, height):
                    draw.point((x, y), fill=self._noise_color())

        # 4. 输出文字:
        x = width / 3 + 1
        y = height / 3 - 1
        # draw.text((x, y), str(number), font=font, fill=self._black_color())
        draw.text((x, y), str(number), fill=self._black_color())

        # 6. 保存文件
        image.save(file_name, format=self.image_format)

    """
    功能：黑色
    参数：NULL
    返回值：RGB 三元组(黑色)
    """

    def _black_color(self):
        return self.black, self.black, self.black

    """
    功能：白色
    参数：NULL
    返回值：RGB 三元组(黑色)
    """

    def _white_color(self):
        return self.white, self.white, self.white

    """
    功能：噪声的颜色
    参数：NULL
    返回值：RGB 三元组(随机灰色)
    """

    def _noise_color(self):
        # 随机数
        noise = random.random() * (self.white - 200) + 200

        # 四舍五入取整
        noise = round(noise)

        return noise, noise, noise


"""
功能：测试 NumberImageSample
参数：NULL 
返回值：NULL
"""


def test():
    nis = NumberImageSample(20, 20, 16, 16)

    nis.create_image(None, 1000)
