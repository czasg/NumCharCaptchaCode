# coding: utf-8
import re
import base64
import numpy as np
import tensorflow as tf

from PIL import Image
from model.cnn.CNN import CNN
from io import BytesIO
from functools import reduce
from trainModel.utils import catchErrorAndRetunDefault
from trainModel.sougou_wechat.gatherCaptcha.gather import Manager


class SouGouWeChat(CNN):
    width = 203  # 图片宽度
    height = 66  # 图片高度
    labelLen = 7

    def __init__(self):
        super(SouGouWeChat, self).__init__()
        self.predictSess = None
        self.gatherManager = None
        self.initPathParams(__file__)

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

    def train(self):
        self.initPath()
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

                valid_x, valid_y = self.get_batch(size=1, test=True)
                textListPre, textListTru = sess.run([pre, tru], feed_dict={
                    self.x: valid_x,
                    self.y: batch_y,
                    self.keepProb: 1.
                })
                print(self.list2text(textListPre), self.list2text(textListTru))

                if index % 10 == 0:
                    acc_image, acc_char = sess.run([imgAccuracy, charAccuracy], feed_dict={
                        self.x: batch_x,
                        self.y: batch_y,
                        self.keepProb: 1.
                    })
                    print(f"图片准确率为 {acc_image: <.5F} - 字符准确率为 {acc_char: <.5F}")

                if index % 500 == 0:
                    self.saver(sess, saver)
            self.saver(sess, saver)

    def saveCaptcha(self, img):
        pass

    def nextCaptcha(self):
        if not self.gatherManager:
            self.gatherManager = Manager()
        return self.gatherManager.nextCaptcha()

    @catchErrorAndRetunDefault
    def predict(self, img):
        if isinstance(img, str):  # base64
            img = base64.b64decode(re.sub('^data:image/.+?;base64,', '', img))
        if isinstance(img, bytes):
            img = Image.open(BytesIO(img))
        imgArray = np.array(img)
        imageGray = self.img2gray(imgArray)
        imageMat = imageGray.flatten() / 255

        if not self.predictSess:
            prediction = self.model()[1]
            pre = tf.argmax(tf.reshape(prediction, [-1, self.labelLen, self.labelSet.__len__()]), 2)
            sess = tf.Session()
            self.predictSess = (sess, pre)
            self.saver(sess)
        sess, pre = self.predictSess
        matList = sess.run(pre, feed_dict={
            self.x: [imageMat],
            self.keepProb: 1.
        })
        return self.list2text(matList)


if __name__ == '__main__':
    # {'abnormal': 0, 'error': 2919, 'right': 24768}
    SouGouWeChat().train()
