# coding: utf-8
from copy import deepcopy

from trainModel.sougou_wechat.SouGouWechat import SouGouWeChat
from trainModel.chuangyu.ChuangYu import ChuangYu
from trainModel.jingdong.JingDong import JingDong
from trainModel.suning.SuNing import SuNing

__all__ = "ModelManager",


class FakerModel:

    def __getattribute__(self, item):
        return lambda *args, **kwargs: {
            "status": 0,
            "msg": "不存在对应的模型",
            "data": None
        }


faker = FakerModel()


class ModelMetaclass(type):

    def __new__(cls, name, bases, attrs: dict):
        if name == "ModelManager":
            attrBacks = deepcopy(attrs)
            attrs.setdefault("__model__", dict())
            for key, value in attrBacks.items():
                if key.startswith("model_"):
                    attr = key.split('_', 1)[1]
                    attrs["__model__"].setdefault(attr, value())
        return super(ModelMetaclass, cls).__new__(cls, name, bases, attrs)


class ModelManager(metaclass=ModelMetaclass):
    model_SouGouWeChat = SouGouWeChat  # 搜狗微信
    model_ChuangYu = ChuangYu  # 知道创宇
    model_JingDong = JingDong  # 京东商城
    model_SuNing = SuNing  # 苏宁易购

    def predict(self, img, model=None):
        return self.__model__.get(model, faker).predict(img)

    def saveCaptcha(self, model, img, code):
        self.__model__.get(model, faker).saveCaptcha(img, code)

    def nextCaptcha(self, model):
        return self.__model__.get(model, faker).nextCaptcha()

    __model__ = dict()
