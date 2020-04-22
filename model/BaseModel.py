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
    curPath = None
    count = 0
    pool = []
    yieldHandler = None

    def __init__(self, curPath):
        self.curPath = curPath
        self.path = os.path.join(self.curPath, self.path)

    def setPath(self, path):
        self.path = os.path.join(self.curPath, path)
        return self.path

    def toPath(self, img):
        return os.path.join(self.path, img)

    def saveCaptcha(self, body: bytes, img: str) -> None:
        with open(self.toPath(img), 'wb') as f:
            f.write(body)

    def setPool(self, pool):
        if isinstance(pool, str):
            self.pool = os.listdir(pool)
        elif isinstance(pool, list):
            self.pool += pool
        random.shuffle(self.pool)

    def yieldBatch(self, pathDir):
        self.setPool(pathDir)
        while True:
            if self.pool:
                for img in self.pool:
                    image = Image.open(self.toPath(img))
                    yield img.split("_")[0], np.array(image)
            else:
                sys.exit(f"{self.name} 数据为空 \n"
                         f"路径: {self.path}")

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

    class ModelPathClass(Path):
        name = "模型文件夹"
        path = "model/"

    class TrainPathClass(Path):
        name = "训练文件夹"
        path = "img/train/"

    class ValidPathClass(Path):
        name = "验证文件夹"
        path = "img/valid/"

    class NewTrainPathClass(Path):
        name = "最新待训练"
        path = "img/trainNew/"

    def __init__(self):
        self.initTensorflow()

    def initTensorflow(self):
        with tf.name_scope('NumCharCC'):
            self.x = tf.compat.v1.placeholder(tf.float32, [None, self.width * self.height])
            self.y = tf.compat.v1.placeholder(tf.float32, [None, self.labelLen * self.labelSet.__len__()])
            self.keepProb = tf.compat.v1.placeholder(tf.float32)

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
        self.ModelPath = self.ModelPathClass(self.curPath)
        self.ValidPath = self.ValidPathClass(self.curPath)
        self.NewTrainPath = self.NewTrainPathClass(self.curPath)
        self.TrainPath = self.TrainPathClass(self.curPath)

    def initPath(self):
        info = ""
        for pathClass in (self.ModelPath, self.ValidPath, self.NewTrainPath, self.TrainPath):
            os.makedirs(pathClass.path, exist_ok=True)
            pathClass.count = len(os.listdir(pathClass.path))
            info += f"{pathClass.name}: {pathClass.count} \n"
            if pathClass.__class__.__name__ == "TrainPath" and pathClass.count == 0:
                sys.exit(f"{info} \nERROR:{pathClass.name} 数据为空")
        if self.ValidPath.count == 0:
            print("检测到测试集为空...以训练集数据作为测试集")
            self.ValidPath.path = self.TrainPath.path
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
                    self.NewTrainPath.toPath(img),
                    self.TrainPath.toPath(img)
                )
            self.TrainPath.setPool(newTrain)
            print("转移成功...")

    def getBatchX(self, imageArray):
        return imageArray.flatten() / 255

    def getBatchY(self, label):
        return self.label2vec(label, self.labelLen, self.labelSet)

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
            batch_x[index, :] = self.getBatchX(imageArray)
            batch_y[index, :] = self.getBatchY(label)
        return batch_x, batch_y

    def saver(self, sess, saver=None):
        if saver:
            print("############## 保存模型 ##############")
            return saver.save(sess, self.ModelPath.path)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, self.ModelPath.path)
            print(f"训练已有模型...{self.ModelPath.path}")
        except:
            print(f"未获取到模型...{self.ModelPath.path}")
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
