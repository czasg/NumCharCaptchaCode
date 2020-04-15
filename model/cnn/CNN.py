# coding: utf-8
import tensorflow as tf
from model.BaseModel import Model
from functools import reduce


class CNN(Model):

    def __init__(self):
        self.w = lambda shape: tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        self.b = lambda shape: tf.Variable(tf.constant(0.1, shape=shape))
        self.co2d = lambda x, w: tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        self.pooling = lambda x: tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        super(CNN, self).__init__()

    def addLayer(self, x, wShape, bShape, *args):
        w = self.w(wShape)
        b = self.b(bShape)
        return self.wxb(w, x, b, *args)

    def addConv(self, x, wShape, bShape):
        """添加`卷积层`
        x:
        wShape:
        bShape:
        """
        w = self.w(wShape)
        b = self.b(bShape)
        wx = self.co2d(x, w)
        y = wx + b
        return self.pooling(
            tf.nn.relu(y)
        )

    def fullConnect(self, x, row, col, output, keep_prob=0.5):
        x = self.addLayer(x, [row, col], [col], tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = self.addLayer(x, [col, output], [output])
        return tf.nn.softmax(x)

    @staticmethod
    def demo():
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=True)  # 获取数据

        cnn = CNN()

        xs = tf.placeholder(tf.float32, [None, 784])  # 输入数据
        ys = tf.placeholder(tf.float32, [None, 10])  # 验证数据
        keep_prob = tf.placeholder("float")  # 过拟合

        img = tf.reshape(xs, [-1, 28, 28, 1])  # 将数据转化为4d向量，其第2、第3维对应图片的宽、高

        conv = cnn.addConv(img, [5, 5, 1, 32], [32])  # 添加卷积层 -> 14 * 14
        conv = cnn.addConv(conv, [5, 5, 32, 64], [64])  # 添加卷积层 -> 7 * 7

        print(f"当前卷积层shape: {conv.shape}")
        length = int(reduce(lambda x, y: x * y, conv.shape[1:]))  # 最后一层卷积的多维数据长度
        x = tf.reshape(conv, [-1, length])  # 多维转化为一维

        prediction = cnn.fullConnect(x, length, 1024, 10, keep_prob)  # 全连接层

        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(ys * tf.log(prediction))
        )  # 交叉熵

        trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))  # 正确预测 -> True, False
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 正确预测 -> 1, 0

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for _ in range(1):
            batch = mnist.train.next_batch(50)
            sess.run(trainStep, feed_dict={
                xs: batch[0],
                ys: batch[1],
                keep_prob: 0.5
            })
            print(sess.run(accuracy, feed_dict={
                xs: mnist.test.images,
                ys: mnist.test.labels,
                keep_prob: 1
            }))


if __name__ == '__main__':
    CNN.demo()
