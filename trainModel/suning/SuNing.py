# coding: utf-8
from model.cnn.CNN import CNN


class SuNing(CNN):
    width = 80  # 图片宽度
    height = 30  # 图片高度
    rotate = True
    labelLen = 4

    def __init__(self):
        super(SuNing, self).__init__()
        self.initPathParams(__file__)


if __name__ == '__main__':
    # SuNing().fastTrain()
    SuNing().keepTrain()
