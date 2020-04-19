# coding: utf-8
import tensorflow as tf

from model.BaseModel import Model


class NN(Model):
    width = 0
    height = 0
    labelLen = 0

    def __init__(self):
        self.w = lambda shape: tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))
        self.b = lambda shape: tf.Variable(tf.constant(0.1, shape=shape))
        super(NN, self).__init__()

    def addLayer(self, x, wShape, bShape, *args):
        w = self.w(wShape)
        b = self.b(bShape)
        return self.wxb(w, x, b, *args)

    @staticmethod
    def demo():
        import numpy as np
        nn = NN()
        x = np.linspace(-1, 1, 300, dtype=np.float)[:, np.newaxis]
        noise = np.random.normal(0, 0.05, x.shape).astype(np.float)
        y = np.square(x) - 0.5 + noise

        xs = tf.placeholder(tf.float32, [None, 1])
        ys = tf.placeholder(tf.float32, [None, 1])
        l1 = nn.addLayer(xs, [1, 10], [1, 10], tf.nn.relu)
        prediction = nn.addLayer(l1, [10, 1], [1, 1])
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

        trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)
        for _ in range(100):
            _, _loss = sess.run([trainStep, loss], feed_dict={
                xs: x,
                ys: y
            })
            print(_loss)


if __name__ == '__main__':
    NN.demo()
