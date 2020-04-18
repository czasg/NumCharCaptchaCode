# coding: utf-8
import os
import re
import sys
import time
import shutil
import random
import base64
import numpy as np
import tensorflow as tf

from PIL import Image
from trainModel.utils import catchErrorAndRetunDefault


class Path:
    name = None
    path = None
    count = 0
    pool = []
    yieldHandler = None

    @classmethod
    def to_path(cls, img):
        return os.path.join(cls.path, img)

    @classmethod
    def saveCaptcha(cls, body: bytes, img: str) -> None:
        with open(cls.to_path(img), 'wb') as f:
            f.write(body)

    @classmethod
    def setPool(cls, pool):
        if isinstance(pool, str):
            cls.pool = os.listdir(pool)
        elif isinstance(pool, list):
            cls.pool += pool
        random.shuffle(cls.pool)

    @classmethod
    def yieldBatch(cls, pathDir):
        cls.setPool(pathDir)
        while True:
            for img in cls.pool:
                image = Image.open(cls.to_path(img))
                yield img.split("_")[0], np.array(image)

    @classmethod
    def nextCaptcha(self):
        if not self.yieldHandler:
            self.yieldHandler = self.yieldBatch(self.path)
        return next(self.yieldHandler)


class Model:
    width = None  # 图片宽度
    height = None  # 图片高度
    labelLen = None  # 标签最大长度
    labelSet = "0123456789abcdefghijklmnopqrstuvwxyz "  # 标签数据集

    stepToSaver = 500  # 保存模型step
    stepToShowAcc = 10  # 打印准确率step
    cycle_loop = 20000  # 训练次数

    class ModelPath:
        name = "模型文件夹"
        path = "model/"

    class TrainPath(Path):
        name = "训练文件夹"
        path = "img/train/"

    class ValidPath(Path):
        name = "验证文件夹"
        path = "img/valid/"

    class NewTrainPath(Path):
        name = "最新待训练"
        path = "img/trainNew/"

    def __init__(self):
        self.initTensorflow()

    def initTensorflow(self):
        with tf.name_scope('sgParams'):
            self.x = tf.placeholder(tf.float32, [None, self.width * self.height])
            self.y = tf.placeholder(tf.float32, [None, self.labelLen * self.labelSet.__len__()])
            self.keepProb = tf.placeholder(tf.float32)

    @staticmethod
    def wxb(w, x, b, activeFunc=lambda x: x):
        wx = tf.matmul(x, w)
        y = wx + b
        return activeFunc(y)

    @staticmethod
    def img2gray(img):
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    @staticmethod
    def label2vec(label, labelLen, labelSet):
        assert len(label) <= labelLen, f"标签长度超过了定义的长度: '{label}'.__len__() > {labelLen}"
        vector = np.zeros(labelLen * len(labelSet))
        for i, ch in enumerate(label):
            idx = i * len(labelSet) + labelSet.index(ch)
            vector[idx] = 1
        return vector

    def list2text(self, predict):
        return "".join(
            filter(
                lambda x: x.strip(),
                (self.labelSet[index] for index in predict[0].tolist())
            )
        )

    def initPathParams(self, filePath=None):
        self.yieldTrainBatchHandler = None
        self.yieldValidBatchHandler = None
        self.curPath = os.path.dirname(os.path.abspath(filePath or __file__))
        for pathClass in (self.ModelPath, self.ValidPath, self.NewTrainPath, self.TrainPath):
            pathClass.path = os.path.join(self.curPath, pathClass.path)

    def initPath(self):
        info = ""
        for pathClass in (self.ModelPath, self.ValidPath, self.NewTrainPath, self.TrainPath):
            os.makedirs(pathClass.path, exist_ok=True)
            pathClass.count = len(os.listdir(pathClass.path))
            info += f"{pathClass.name}: {pathClass.count} \n"
            if pathClass.__name__ == "TrainPath" and pathClass.count == 0:
                sys.exit(f"{info} \nERROR:{pathClass.name} 数据为空")
        print(info)

    @catchErrorAndRetunDefault
    def saveCaptcha(self, img, code):
        if isinstance(img, str):
            img = base64.b64decode(re.sub('^data:image/.+?;base64,', '', img))
        if isinstance(img, bytes):
            filename = f"{code}_{int(time.time())}.jpg"
            self.NewTrainPath.saveCaptcha(img, filename)
            return {
                "status": 0,
                "msg": f"保存成功: {filename}"
            }
        else:
            return {
                "status": 0,
                "msg": f"保存失败, 未知格式: {type(img)}"
            }

    def checkNewTrainPath(self):
        newTrain = os.listdir(self.NewTrainPath.path)
        if newTrain:
            print(f"检测到新的训练数据: {len(newTrain)} 开始转移数据...")
            for img in newTrain:
                shutil.move(
                    self.NewTrainPath.to_path(img),
                    self.TrainPath.to_path(img)
                )
            self.TrainPath.setPool(newTrain)
            print("转移成功...")

    def get_batch(self, test=False, size=100):
        self.checkNewTrainPath()
        batch_x = np.zeros([size, self.height * self.width])
        batch_y = np.zeros([size, self.labelLen * self.labelSet.__len__()])
        for index in range(size):
            if test:
                label, imageArray = self.ValidPath.nextCaptcha()
            else:
                label, imageArray = self.TrainPath.nextCaptcha()

            offset = self.labelLen - len(label)
            if offset > 0:
                label += ' ' * offset
            elif offset < 0:
                continue

            imageArray = self.img2gray(imageArray)
            batch_x[index, :] = imageArray.flatten() / 255
            batch_y[index, :] = self.label2vec(label, self.labelLen, self.labelSet)
        return batch_x, batch_y

    def saver(self, sess, saver=None):
        print(self.ModelPath.path)
        if saver:
            print("保存模型...")
            return saver.save(sess, self.ModelPath.path)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, self.ModelPath.path)
            print("训练已有模型...")
        except:
            print("未获取到模型...")
        return saver

    def valid(self, prediction):
        predict = tf.reshape(prediction, [-1, self.labelLen, self.labelSet.__len__()])
        truth = tf.reshape(self.y, [-1, self.labelLen, self.labelSet.__len__()])
        pre = tf.argmax(predict, 2)
        tru = tf.argmax(truth, 2)
        correctPrediction = tf.equal(pre, tru)
        charAccuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        imgAccuracy = tf.reduce_mean(tf.reduce_min(tf.cast(correctPrediction, tf.float32), axis=1))
        return pre, tru, charAccuracy, imgAccuracy
