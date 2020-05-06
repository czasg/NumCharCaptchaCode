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
from trainModel.utils import catchErrorAndRetunDefault, show_dynamic_ratio


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
        path = "model/"  # 需要以斜杠结尾

    class TrainPathClass(Path):
        name = "训练文件夹"
        path = "img/train"

    class ValidPathClass(Path):
        name = "验证文件夹"
        path = "img/valid"

    class NewTrainPathClass(Path):
        name = "最新待训练"
        path = "img/trainNew/"

    def __init__(self):
        self.initTensorflow()

    def initTensorflow(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
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
        if self.TrainPath.count == 0 and self.NewTrainPath.count == 0:
            sys.exit(f"{info} \nERROR:{pathClass.name} 数据为空")
        if self.ValidPath.count == 0:
            print("检测到测试集为空...以训练集数据作为测试集")
            self.ValidPath.path = self.TrainPath.path
        print(info)

    def checkNewTrainPath(self, size=1):
        newTrain = os.listdir(self.NewTrainPath.path)
        if len(newTrain) > size:
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
        self.checkNewTrainPath(size)
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

    def keep_batch(self, batch_x: np.array, batch_y: np.array, preRight: np.array):
        self.checkNewTrainPath(preRight.__len__())
        count = 0
        keepRate = np.sum(preRight) / preRight.__len__()
        for index, value in enumerate(preRight):
            if value == 1.0:
                pass
            elif random.random() < keepRate:
                count += 1
                continue
            label, imageArray = self.TrainPath.nextCaptcha()

            offset = self.labelLen - len(label)
            if offset > 0:
                label += ' ' * offset
            elif offset < 0:
                continue

            imageArray = self.img2gray(imageArray)
            batch_x[index, :] = self.getBatchX(imageArray)
            batch_y[index, :] = self.getBatchY(label)
        print(f">>> 图片准确率: {keepRate: <.3F} - 保留率为: {count}/{preRight.__len__()}")
        return batch_x, batch_y

    def testStepShow(self, sess, pre, tru):
        valid_x, valid_y = self.get_batch(size=1, test=True)
        textListPre, textListTru = sess.run([pre, tru], feed_dict={
            self.x: valid_x,
            self.y: valid_y,
            self.keepProb: 1.
        })
        print(f"测试集 >>> 预测数据: {self.list2text(textListPre)}  实际数据: {self.list2text(textListTru)}")

    def trainStepShow(self, sess, batch_x, batch_y, imgAccuracy, charAccuracy):
        acc_image, acc_char = sess.run([imgAccuracy, charAccuracy], feed_dict={
            self.x: batch_x,
            self.y: batch_y,
            self.keepProb: 1.
        })
        print(f"训练集 >>> 图片准确率: {acc_image} - 字符准确率: {acc_char}")

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
        pre = tf.argmax(predict, 2)  # 预测值
        tru = tf.argmax(truth, 2)  # 实际值
        correctPrediction = tf.equal(pre, tru)
        charAccuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))  # 单个字符匹配准确率
        imgAccuracy = tf.reduce_sum(tf.reduce_min(tf.cast(correctPrediction, tf.float32), axis=1))  # 全匹配准确率
        listAccuracy = tf.reduce_min(tf.cast(correctPrediction, tf.float32), axis=1)  # 匹配值列表
        return pre, tru, charAccuracy, imgAccuracy, listAccuracy

    def checkTrained(self, sess, pre):
        print("正在检测已训练数据中是否有需要重新训练的...")
        self.predictSess = (sess, pre)
        allCount = errorCount = index = 0
        while True:
            trainedPath = f"{self.TrainPath.path}_{index}"
            if os.path.exists(trainedPath):
                allTrained = os.listdir(trainedPath)
                allTrainedLenght = allTrained.__len__()
                print(f">>> 检测到已训练数据: {allTrainedLenght} >>> 路径: {trainedPath}")
                count = 0
                for img in allTrained:
                    imgPath = os.path.join(trainedPath, img)
                    with open(imgPath, "rb") as f:
                        body = f.read()
                    if img.split("_")[0] != self.predict(body):
                        shutil.move(
                            imgPath,
                            self.NewTrainPath.toPath(img)
                        )
                        errorCount += 1
                    allCount += 1
                    count += 1
                    show_dynamic_ratio(count, allTrainedLenght)
                if count: print("")
            else:
                break
            index += 1
        print(f">>> 已训练数据总量:{allCount} \n>>> 需要重新训练量:{errorCount}")

    def saveTrained(self):
        index = 0
        while True:
            trainedPath = f"{self.TrainPath.path}_{index}"
            if os.path.exists(trainedPath):
                index += 1
                continue
            os.rename(self.TrainPath.path, trainedPath)
            os.makedirs(self.TrainPath.path)
            print(f">>> 此次训练数据已转到: {trainedPath}")
            break

    def predict(self, img) -> str:
        raise NotImplementedError

    @catchErrorAndRetunDefault
    def saveCaptcha(self, img, code) -> dict:
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

    def nextCaptcha(self, code=None, base64str=None) -> dict:
        return {
            "status": 1,
            "msg": "获取成功",
            "data": {
                "code": code,
                "image": f"data:image/png;base64,{base64str}"
            }
        }
