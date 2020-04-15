import tensorflow as tf


class Model:
    width = None
    height = None

    def __init__(self):
        pass

    def wxb(self, w, x, b, activeFunc=lambda x: x):
        return activeFunc(tf.matmul(x, w) + b)
