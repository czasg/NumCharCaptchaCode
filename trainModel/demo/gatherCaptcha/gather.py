import os
import time
import random
import requests
from itertools import count
from minitools.javascript import get_anti_spider_clearance
from captcha.image import ImageCaptcha


class Captchar:

    def __init__(self):
        self.width = 200
        self.height = 60
        self.code = lambda: "".join(random.sample('0123456789abcdefghijklmnopqrstuvwxyz', 6))

    def saveCaptcha(self):
        code = self.code()
        ImageCaptcha(width=self.width, height=self.height).generate_image(code).save(
            f"../img/valid/{code}_{int(time.time())}.png")

    def nextCaptcha(self):
        return ImageCaptcha(width=self.width, height=self.height).generate_image(self.code()).tobytes()

    def loopCaptcha(self, loop: int = 100, delay=0):
        while loop:
            self.nextCaptcha()
            loop -= 1
            time.sleep(delay)


class Manager:

    def __init__(self):
        self.captcha = Captchar()

    def run(self):
        count = 1000
        while count:
            self.captcha.saveCaptcha()
            count -= 1

    def nextCaptcha(self):
        return self.captcha.nextCaptcha()


if __name__ == '__main__':
    Manager().run()
