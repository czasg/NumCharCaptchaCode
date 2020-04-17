# coding: utf-8
import os
import random
import numpy as np
import tensorflow as tf

from PIL import Image


class Model:
    width = None  # 图片宽度
    height = None  # 图片高度
    labelLen = None
    labelSet = "0123456789abcdefghijklmnopqrstuvwxyz "  # 标签数据集
    modelPath = "model/"
    trainPath = "img/train/"
    validPath = "img/valid/"
    newTrainPath = "img/trainNew/"

    cycle_loop = 20000
    stillAccToStop = 0.99

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
        text = ""
        for index in predict[0].tolist():
            text += str(self.labelSet[index])
        return text

    def initPathParams(self, filePath=None):
        self.yieldTrainBatchHandler = None
        self.yieldValidBatchHandler = None
        self.curPath = os.path.dirname(os.path.abspath(filePath or __file__))
        self.filePath = lambda path, img: os.path.join(self.curPath, path, img)

    def initPath(self):
        info = ""
        for path in [self.modelPath, self.trainPath, self.validPath, self.newTrainPath]:
            path = os.path.join(self.curPath, path)
            os.makedirs(path, exist_ok=True)
            info += f"{path} 文件量: {len(os.listdir(path))} \n"
        print(info)

    def yieldBatch(self, pathDir):
        pool = os.listdir(pathDir)
        random.shuffle(pool)
        while True:
            for img in pool:
                image = Image.open(self.filePath(pathDir, img))
                yield img.split("_")[0], np.array(image)

    def yieldTrainBatch(self):
        if not self.yieldTrainBatchHandler:
            self.yieldTrainBatchHandler = self.yieldBatch(self.trainPath)
        return next(self.yieldTrainBatchHandler)

    def yieldValidBatch(self):
        if not self.yieldValidBatchHandler:
            self.yieldValidBatchHandler = self.yieldBatch(self.validPath)
        return next(self.yieldValidBatchHandler)

    def get_batch(self, test=False, size=100):
        batch_x = np.zeros([size, self.height * self.width])
        batch_y = np.zeros([size, self.labelLen * self.labelSet.__len__()])
        for index in range(size):
            if test:
                label, imageArray = self.yieldValidBatch()
            else:
                label, imageArray = self.yieldTrainBatch()

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
        if saver:
            print("保存模型...")
            return saver.save(sess, os.path.join(self.curPath, self.modelPath))
        saver = tf.train.Saver()
        try:
            saver.restore(sess, os.path.join(self.curPath, self.modelPath))
            print("训练已有模型...")
        except:
            print("创建新模型...")
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
