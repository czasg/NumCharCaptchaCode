# coding: utf-8
import os
import re
import random
import numpy as np
import tensorflow as tf

from PIL import Image
from model.cnn.CNN import CNN
from io import BytesIO, StringIO
from functools import partial, reduce


class SouGouWechat(CNN):
    width = 203  # 图片宽度
    height = 66  # 图片高度
    labelLen = 7
    labelSet = "0123456789abcdefghijklmnopqrstuvwxyz "  # 标签数据集
    modelPath = "model/"
    trainPath = "img/train/"
    validPath = "img/valid/"
    newTrainPath = "img/trainNew/"

    cycle_loop = 20000
    stillAccToStop = 0.99

    def __init__(self):
        self.linear = partial(self.label2vec, labelLen=self.labelLen, labelSet=self.labelSet)
        self.initPath()
        self.initTensorflow()
        super(SouGouWechat, self).__init__()

        self.yieldTrainBatchHandler = None
        self.yieldValidBatchHandler = None
        self.predictSess = None

    def initPath(self):
        curPath = os.path.dirname(os.path.abspath(__file__))
        self.filePath = lambda path, img: os.path.join(curPath, path, img)
        info = ""
        for path in [self.modelPath, self.trainPath, self.validPath, self.newTrainPath]:
            os.makedirs(path, exist_ok=True)
            info += f"{path} 文件量: {len(os.listdir(path))} \n"
        print(info)

    def initTensorflow(self):
        with tf.name_scope('sgParams'):
            self.x = tf.placeholder(tf.float32, [None, self.width * self.height])
            self.y = tf.placeholder(tf.float32, [None, self.labelLen * self.labelSet.__len__()])
            self.keepProb = tf.placeholder(tf.float32)

    def model(self):
        img = tf.reshape(self.x, [-1, self.height, self.width, 1])  # 1d -> 4d

        conv = self.addConv(img, [3, 3, 1, 32], [32])  # 卷积层 1
        conv = self.addConv(conv, [3, 3, 32, 64], [64])  # 卷积层 2
        conv = self.addConv(conv, [3, 3, 64, 128], [128])  # 卷积层 3

        length = int(reduce(lambda x, y: x * y, conv.shape[1:]))
        x = tf.reshape(conv, [-1, length])  # 4d -> 1d

        prediction = self.fullConnect(
            x, length, 1024, self.labelLen * self.labelSet.__len__(), self.keepProb
        )  # 全连接层

        with tf.name_scope('cost'):
            crossEntropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=self.y)
            )
        with tf.name_scope('train'):
            trainStep = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)
        return trainStep, prediction

    def valid(self, prediction):
        predict = tf.reshape(prediction, [-1, self.labelLen, self.labelSet.__len__()])
        truth = tf.reshape(self.y, [-1, self.labelLen, self.labelSet.__len__()])
        pre = tf.argmax(predict, 2)
        tru = tf.argmax(truth, 2)

        correctPrediction = tf.equal(pre, tru)

        charAccuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        imgAccuracy = tf.reduce_mean(tf.reduce_min(tf.cast(correctPrediction, tf.float32), axis=1))
        return pre, tru, charAccuracy, imgAccuracy

    def saver(self, sess, saver=None):
        if saver:
            print("保存模型...")
            return saver.save(sess, self.modelPath)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, self.modelPath)
            print("训练已有模型...")
        except:
            print("创建新模型...")
        return saver

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
            batch_y[index, :] = self.linear(label)
        return batch_x, batch_y

    def list2text(self, predict):
        text = ""
        for index in predict[0].tolist():
            text += str(self.labelSet[index])
        return text

    def train(self):
        trainStep, prediction = self.model()
        pre, tru, charAccuracy, imgAccuracy = self.valid(prediction)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = self.saver(sess)

            for index in range(self.cycle_loop):
                batch_x, batch_y = self.get_batch()

                sess.run(trainStep, feed_dict={
                    self.x: batch_x,
                    self.y: batch_y,
                    self.keepProb: 0.5
                })

                valid_x, valid_y = self.get_batch(size=1)
                textListPre, textListTru = sess.run([pre, tru], feed_dict={
                    self.x: valid_x,
                    self.y: batch_y,
                    self.keepProb: 1.
                })
                print(self.list2text(textListPre), self.list2text(textListTru))

                if index % 10 == 0:
                    # valid_x, valid_y = self.get_batch(test=True)
                    acc_image, acc_char = sess.run([imgAccuracy, charAccuracy], feed_dict={
                        self.x: batch_x,
                        self.y: batch_y,
                        self.keepProb: 1.
                    })
                    print(f"图片准确率为 {acc_image: <.5F} - 字符准确率为 {acc_char: <.5F}")

                if index % 500 == 0:
                    self.saver(sess, saver)
            self.saver(sess, saver)

    def predict(self, img):
        if isinstance(img, str):  # base64
            img = Image.open(StringIO(re.sub('^data:image/.+?;base64,', '', img)))
        elif isinstance(img, bytes):
            img = Image.open(BytesIO(img))
        imgArray = np.array(img)
        imageGray = self.img2gray(imgArray)
        imageMat = imageGray.flatten() / 255

        if not self.predictSess:
            prediction = self.model()[1]
            pre = tf.argmax(tf.reshape(prediction, [-1, self.labelLen, self.labelSet.__len__()]), 2)
            self.predictSess = (tf.Session(), pre)
            self.saver(self.predictSess)
        sess, pre = self.predictSess
        matList = sess.run(pre, feed_dict={
            self.x: [imageMat],
            self.keepProb: 1.
        })
        return self.list2text(matList)

    def predictTest(self, img):
        if isinstance(img, str):  # base64
            img = Image.open(StringIO(re.sub('^data:image/.+?;base64,', '', img)))
        elif isinstance(img, bytes):
            img = Image.open(BytesIO(img))
        imgArray = np.array(img)
        imageGray = self.img2gray(imgArray)
        imageMat = imageGray.flatten() / 255

        prediction = self.model()[1]
        pre = tf.argmax(tf.reshape(prediction, [-1, self.labelLen, self.labelSet.__len__()]), 2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver(sess)
            matList = sess.run(pre, feed_dict={
                self.x: [imageMat],
                self.keepProb: 1.
            })
            return self.list2text(matList)

    def test(self):
        img = Image.open('./img/valid/2a6bd8_1576634610.jpg')
        print(self.predictTest(img))


if __name__ == '__main__':
    SouGouWechat().train()

    # SouGouWechat().test()
