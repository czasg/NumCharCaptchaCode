import tensorflow as tf
from model.BaseModel import Model


class CNN(Model):

    def __init__(self):
        self.w = lambda shape: tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        self.b = lambda shape: tf.Variable(tf.constant(0.1, shape=shape))
        self.co2d = lambda x, w: tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        self.pooling = lambda x: tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    # def createW(self, shape):
    #     return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    #
    # def createB(self, shape):
    #     return tf.Variable(tf.constant(0.1, shape=shape))
    #
    # def to2d(self, x, w):
    #     return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    #
    # def max_pooling(self, x):
    #     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

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
        return self.pooling(self.addLayer(x, wShape, bShape, tf.nn.relu))

    def fullConnect(self, x, row, col, output, drop=0.5):
        x = self.addLayer(x, [row, col], [col], tf.nn.relu)
        x = tf.nn.drop(x, (1 - drop))
        x = self.addLayer(x, [col, output], [output])
        return tf.nn.softmax(x)

    @staticmethod
    def demo():
        cnn = CNN()
        xs = tf.placeholder(tf.float32, [None, 784])
        ys = tf.placeholder(tf.float32, [None, 10])
        img = tf.reshape(xs, [-1, 28, 28, 1])
        conv1 = cnn.addConv(img, [5, 5, 1, 32], [32])
        conv2 = cnn.addConv(conv1, [5, 5, 32, 64], [64])
        length = 7 * 7 * 64
        x = tf.reshape(conv2, [-1, length])
        prediction = cnn.fullConnect(x, length, 1024, 10)

        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])
        )  # 交叉熵

        trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(trainStep)


if __name__ == '__main__':
    pass
