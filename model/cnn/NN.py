import tensorflow as tf
from model.BaseModel import Model


class NN(Model):

    def __init__(self):
        pass

    def add_layer(self, x, inSize, outSize, activeFunc=lambda x: x):
        """model-layer:
        y = Wx + b
        """
        w = tf.Variable(tf.random_normal([inSize, outSize]))
        b = tf.Variable(tf.zeros([1, outSize]) + 0.1)
        y = tf.matmul(x, w) + b
        return activeFunc(y)

    @staticmethod
    def demo():
        import numpy as np
        nn = NN()
        x = np.linspace(-1, 1, 300, dtype=np.float)[:, np.newaxis]
        noise = np.random.normal(0, 0.05, x.shape).astype(np.float)
        y = np.square(x) - 0.5 + noise

        xs = tf.placeholder(tf.float32, [None, 1])
        ys = tf.placeholder(tf.float32, [None, 1])
        l1 = nn.add_layer(xs, 1, 10, tf.nn.relu)
        prediction = nn.add_layer(l1, 10, 1)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

        trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)
        for _ in range(200):
            _, _loss = sess.run([trainStep, loss], feed_dict={
                xs: x,
                ys: y
            })
            print(_loss)


if __name__ == '__main__':
    NN.demo()
