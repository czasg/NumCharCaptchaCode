import os
import time
import random
import requests
from itertools import count
from minitools.javascript import get_anti_spider_clearance


class Captchar:

    def __init__(self, headers=None):
        self.headers = headers or {}
        self.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/"
                                           "537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36"})
        self.counter = count().__next__
        self.timestamp = lambda digits=1: int(time.time() * digits)
        self.sess = requests.session()
        self.createSalt = lambda: ''.join(random.sample('987654321zyxwvutsrqponmlkjihgfedcba', 6))
        self.index = "http://www.gsxt.gov.cn/index"
        self.create_uri = lambda salt: f"http://www.gsxt.gov.cn/cdn-cgi/captcha/{salt}/"
        self.savePath = ""
        self.cookies = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def initCookies(self):
        jString = self.sess.get(self.index, headers=self.headers).text
        cookies = get_anti_spider_clearance(jString, self.index)
        key, value = cookies.split('=')
        self.cookies = {key: value}

    def initSavePath(self, path="./save"):
        self.savePath = path
        os.makedirs(path, exist_ok=True)

    def saveCaptcha(self, body):
        if self.savePath:
            with open(f"{self.savePath}/{self.counter()}_{self.timestamp()}.jpg", 'wb') as f:
                f.write(body)

    @classmethod
    def from_headers(cls, headers):
        return cls(headers)

    def nextCaptcha(self):
        if not self.cookies:
            self.initCookies()
        salt = self.createSalt()
        uri = self.create_uri(salt)
        body = self.sess.get(f"{uri}/1", headers=self.headers, cookies=self.cookies).content
        self.saveCaptcha(body)
        return body

    def loopCaptcha(self, loop: int = 100, delay=0):
        while loop:
            self.nextCaptcha()
            loop -= 1
            time.sleep(delay)


class Manager:

    def __init__(self):
        self.captcha = Captchar()

    def run(self, loop=10, delay=0):
        self.captcha.initSavePath()
        self.captcha.loopCaptcha(loop=loop, delay=delay)

    def nextCaptcha(self):
        return self.captcha.nextCaptcha()


if __name__ == '__main__':
    Manager().run()
