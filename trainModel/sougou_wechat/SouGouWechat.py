# coding: utf-8
import base64

from model.cnn.CNN import CNN
from trainModel.utils import catchErrorAndRetunDefault
from trainModel.sougou_wechat.gatherCaptcha.gather import Manager


class SouGouWeChat(CNN):
    width = 203  # 图片宽度
    height = 66  # 图片高度
    labelLen = 7

    def __init__(self):
        super(SouGouWeChat, self).__init__()
        self.gatherManager = None
        self.initPathParams(__file__)

    @catchErrorAndRetunDefault
    def nextCaptcha(self) -> dict:
        if not self.gatherManager:
            self.gatherManager = Manager()
        body = self.gatherManager.nextCaptcha()  # type: bytes
        code = self.predict(body)
        return super(SouGouWeChat, self).nextCaptcha(
            code,
            base64.b64encode(body).decode()
        )


if __name__ == '__main__':
    # SouGouWeChat().fastTrain()
    SouGouWeChat().keepTrain()
