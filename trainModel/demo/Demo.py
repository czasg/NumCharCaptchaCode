# coding: utf-8
from model.cnn.CNN import CNN


class Demo(CNN):
    # width = 200  # 图片宽度
    # height = 60  # 图片高度
    labelLen = 6

    def __init__(self):
        super(Demo, self).__init__()
        self.initPathParams(__file__)

    def defineConv(self, img):
        conv = self.addConv(img, [3, 3, 1, 6], [6])
        conv = self.addConv(conv, [3, 3, 6, 16], [16])
        return conv


if __name__ == '__main__':
    # Demo().fastTrain()
    Demo().keepTrain()
