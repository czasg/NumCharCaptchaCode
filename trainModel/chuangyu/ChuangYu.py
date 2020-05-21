# coding: utf-8
import base64

from model.cnn.CNN import CNN
from trainModel.utils import catchErrorAndRetunDefault


class ChuangYu(CNN):
    width = 139  # 图片宽度
    height = 60  # 图片高度
    labelLen = 6
    rotate = [-3, 3, -6, 6]

    def __init__(self):
        super(ChuangYu, self).__init__()
        self.gatherManager = None
        self.initPathParams(__file__)

    def getBatchX(self, imageArray):
        return imageArray.flatten() + 0

    @catchErrorAndRetunDefault
    def nextCaptcha(self) -> dict:
        if not self.gatherManager:
            self.gatherManager = Manager()
        body = self.gatherManager.nextCaptcha()
        code = self.predict(body)
        return {
            "status": 1,
            "msg": "获取成功",
            "data": {
                "code": code,
                "image": f"data:image/png;base64,{base64.b64encode(body).decode()}"
            }
        }


if __name__ == '__main__':
    # ChuangYu().fastTrain()
    ChuangYu().keepTrain()
