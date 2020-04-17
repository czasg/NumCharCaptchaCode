import json
import logging
import tornado.ioloop
import tornado.web

from tornado.options import define, options

from trainModel import ModelManager

define('port', default=8866, type=int)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
model = ModelManager()


class CaptchaHandler(tornado.web.RequestHandler):

    def post(self):
        """
        input:
        >>> {
        >>>     "model": "SouGouWeChat",
        >>>     "image": "base64"
        >>> }
        output:
        >>> {
        >>>     "status": "-1 / 0 / 1",
        >>>     "msg": "string",
        >>>     "data": "None / code"
        >>> }
        """
        json_data = json.loads(self.request.body)  # type: dict
        img = json_data.get("image")
        if not img:
            return self.write({
                "status": -1,
                "msg": "请携带`image`数据访问",
                "data": None
            })
        code = model.predict(img, json_data.get("model"))  # 图像识别
        if isinstance(code, dict):
            result = code  # 识别失败，返回错误信息
        else:
            result = {
                "status": 1,
                "msg": "识别成功",
                "data": code
            }
        self.write(result)


def run_app():
    app = tornado.web.Application([
        ('/captcha', CaptchaHandler)
    ])
    app.listen(port=options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    run_app()
