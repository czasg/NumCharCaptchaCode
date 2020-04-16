# coding: utf-8
import numpy as np
import tensorflow as tf


class Model:

    def __init__(self):
        pass

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


if __name__ == '__main__':
    print(Model.label2vec('cz ', 3, "0123456789abcdefghijklmnopqrstuvwxyz "))
