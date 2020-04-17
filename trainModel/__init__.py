# coding: utf-8
from copy import deepcopy

from trainModel.sougou_wechat.SouGouWechat import SouGouWeChat

__all__ = "ModelManager",


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

    class FakerModel:
        @staticmethod
        def predict(*args, **kwargs):
            return {
                "status": 0,
                "msg": "不存在对应的模型",
                "data": None
            }

    def predict(self, img, model=None):
        return self.__model__.get(model, self.FakerModel).predict(img)
