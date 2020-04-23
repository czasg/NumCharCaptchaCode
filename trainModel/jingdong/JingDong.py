# coding: utf-8
from model.cnn.CNN import CNN


class JingDong(CNN):
    width = 85  # 图片宽度
    height = 26  # 图片高度
    labelLen = 4

    def __init__(self):
        super(JingDong, self).__init__()
        self.initPathParams(__file__)


if __name__ == '__main__':
    # JingDong().fastTrain()
    JingDong().keepTrain()
