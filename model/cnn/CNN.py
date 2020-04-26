# coding: utf-8
import base64
import numpy as np
import tensorflow as tf

from PIL import Image
from io import BytesIO
from functools import reduce
from model.BaseModel import Model
from trainModel.utils import catchErrorAndRetunDefault, BASE64_REGEX


class CNN(Model):

    def __init__(self):
        self.predictSess = None
        self.w = lambda shape: tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))
        self.b = lambda shape: tf.Variable(tf.constant(0.1, shape=shape))
        self.co2d = lambda x, w: tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        self.pooling = lambda x: tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        super(CNN, self).__init__()

    def addLayer(self, x, wShape, bShape, *args):
        w = self.w(wShape)
        b = self.b(bShape)
        return self.wxb(w, x, b, *args)

    def addConv(self, x, wShape, bShape, dropout=True):
        """添加`卷积层`
        x:
        wShape:
        bShape:
        dropout:
        """
        w = self.w(wShape)
        b = self.b(bShape)
        wx = self.co2d(x, w)
        y = wx + b
        conv = self.pooling(
            tf.nn.relu(y)
        )
        return tf.nn.dropout(conv, rate=(1 - self.keepProb)) \
            if dropout else \
            conv

    def fullConnect(self, x, row, col, output):
        x = self.addLayer(x, [row, col], [col], tf.nn.relu)
        x = tf.nn.dropout(x, rate=(1 - self.keepProb))
        x = self.addLayer(x, [col, output], [output])
        return x

    def defineConv(self, img):
        """
        简单验证码卷积层参考 1 -> 6 -> 16
        >>> conv = self.addConv(img, [3, 3, 1, 6], [6])
        >>> conv = self.addConv(conv, [3, 3, 6, 16], [16])
        复杂验证码卷积层参考 1 -> 32 -> 64 -> 128
        >>> conv = self.addConv(img, [3, 3, 1, 32], [32])
        >>> conv = self.addConv(conv, [3, 3, 32, 64], [64])
        >>> conv = self.addConv(conv, [3, 3, 64, 128], [128])
        :param img:
        :return:
        """
        conv = self.addConv(img, [3, 3, 1, 32], [32])  # 卷积层 1
        conv = self.addConv(conv, [3, 3, 32, 64], [64])  # 卷积层 2
        conv = self.addConv(conv, [3, 3, 64, 128], [128])  # 卷积层 3
        return conv

    def model(self):
        img = tf.reshape(self.x, [-1, self.height, self.width, 1])  # 1d -> 4d

        conv = self.defineConv(img)  # 定义卷积层

        length = int(reduce(lambda x, y: x * y, conv.shape[1:]))
        x = tf.reshape(conv, [-1, length])  # 4d -> 1d

        prediction = self.fullConnect(
            x, length, 1024, self.labelLen * self.labelSet.__len__()
        )  # 全连接层

        with tf.name_scope('cost'):
            crossEntropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=self.y)
            )  # 交叉熵

        with tf.name_scope('train'):
            trainStep = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(crossEntropy)
        return trainStep, prediction

    def chooseModel(self):
        return self.model()

    def fastTrain(self):
        self.initPath()
        trainStep, prediction = self.chooseModel()
        pre, tru, charAccuracy, imgAccuracy, _ = self.valid(prediction)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = self.saver(sess)

            for index in range(self.cycle_loop):
                batch_x, batch_y = self.get_batch()
                sess.run(trainStep, feed_dict={
                    self.x: batch_x,
                    self.y: batch_y,
                    self.keepProb: 0.75
                })

                self.testStepShow(sess, pre, tru)
                if index % self.stepToShowAcc == 0:
                    self.trainStepShow(sess, batch_x, batch_y, imgAccuracy, charAccuracy)

                if index % self.stepToSaver == 0:
                    self.saver(sess, saver)
            self.saver(sess, saver)

    def keepTrain(self):
        self.initPath()
        trainStep, prediction = self.chooseModel()
        pre, tru, charAccuracy, imgAccuracy, listAccuracy = self.valid(prediction)
        batch_x, batch_y = self.get_batch()

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = self.saver(sess)

            for index in range(self.cycle_loop):
                _, listAcc = sess.run([trainStep, listAccuracy], feed_dict={
                    self.x: batch_x,
                    self.y: batch_y,
                    self.keepProb: 0.75
                })

                if index % self.stepToShowAcc == 0:
                    self.trainStepShow(sess, batch_x, batch_y, imgAccuracy, charAccuracy)

                batch_x, batch_y = self.keep_batch(batch_x, batch_y, listAcc)

                if index % self.stepToSaver == 0:
                    self.saver(sess, saver)
            self.saver(sess, saver)
            # self.saveTrained()
            # self.checkTrained(sess, pre)

    @catchErrorAndRetunDefault
    def predict(self, img):
        if isinstance(img, str):  # base64
            img = base64.b64decode(BASE64_REGEX("", img))
        if isinstance(img, bytes):
            img = Image.open(BytesIO(img))
            img = img.resize((self.width, self.height))
        imgArray = np.array(img)
        imageGray = self.img2gray(imgArray)

        if not self.predictSess:
            prediction = self.model()[1]
            pre = self.valid(prediction)[0]
            sess = tf.compat.v1.Session()
            sess.run(tf.compat.v1.global_variables_initializer())
            self.predictSess = (sess, pre)
            self.saver(sess)
        sess, pre = self.predictSess
        matList = sess.run(pre, feed_dict={
            self.x: [self.getBatchX(imageGray)],
            self.keepProb: 1.
        })
        return self.list2text(matList)
