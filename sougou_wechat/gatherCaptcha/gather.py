import os
import time
import requests
from itertools import count


class Captchar:

    def __init__(self, headers=None):
        self.headers = headers or {}
        self.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/"
                                           "537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36"})
        self.counter = count().__next__
        self.timestamp = lambda digits=1: int(time.time() * digits)
        self.uri = f"https://weixin.sogou.com/antispider/util/seccode.php?tc={self.timestamp(1000)}"
        self.savePath = ""

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
        body = requests.get(self.uri, headers=self.headers).content
        self.saveCaptcha(body)
        return body

    def loopCaptcha(self, loop: int = 100, delay=0):
        while loop:
            self.nextCaptcha()
            loop -= 1
            time.sleep(delay)


class Manager:

    def __init__(self):
        headers = {
            'accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9,und;q=0.8,en;q=0.7',
            'cookie': 'IPLOC=CN4201; SUID=F6E5EADD4018960A000000005E22B447; '
                      'SUID=F6E5EADD3118960A000000005E22B4F9; weixinIndexVisited=1; '
                      'SUV=006BCFF0DDEAE5F65E22B4FA88FC0224; '
                      'CXID=74064148C841C84596F2DF899D87B306; ABTEST=1|1585812549|v1; '
                      'SNUID=1C179AAF7174D09A28823A99722FB27A; '
                      'JSESSIONID=aaal_OBEmLOie9iGIkJfx; sct=6; '
                      'PHPSESSID=8ssdgmln0sc1vd5lp9l1pcuqc5; refresh=1',
            'referer': 'https://weixin.sogou.com/antispider/?from=%2Fweixin%3Ftype%3D2%26s_from%3Dinput%26query%3DoI'
                       'WsFt6d6SGf6AkN5RSl-swaNWzY%26ie%3Dutf8%26_sug_%3Dn%26_sug_type_%3D%26page%3D1',
            'sec-fetch-dest': 'image',
            'sec-fetch-mode': 'no-cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'
        }
        self.captcha = Captchar.from_headers(headers)

    def run(self, loop=10, delay=0):
        self.captcha.initSavePath()
        self.captcha.loopCaptcha(loop=loop, delay=delay)

    def nextCaptcha(self):
        return self.captcha.nextCaptcha()


if __name__ == '__main__':
    Manager().run()
