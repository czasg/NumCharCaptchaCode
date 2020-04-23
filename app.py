import json
import logging
import tornado.ioloop
import tornado.web

from tornado.options import define, options

from trainModel.ModelManager import ModelManager

define('port', default=8866, type=int)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
model = ModelManager()


class CaptchaHandler(tornado.web.RequestHandler):

    async def post(self):
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
        code = model.predict(img, json_data.get("model"))
        if isinstance(code, dict):
            result = code  # 识别失败，返回错误信息
        else:
            result = {
                "status": 1,
                "msg": "识别成功",
                "data": code
            }
        self.write(result)


class SaveCaptchaHandler(tornado.web.RequestHandler):

    async def post(self):
        """
        input:
        >>> {
        >>>     "model": "SouGouWeChat",
        >>>     "image": "base64",
        >>>     "code": "string",
        >>> }
        output:
        >>> {
        >>>     "status": "-1 / 0 / 1",
        >>>     "msg": "string"
        >>> }
        """
        json_data = json.loads(self.request.body)  # type: dict
        result = await tornado.ioloop.IOLoop.current().run_in_executor(
            None,
            model.saveCaptcha,
            json_data.get("model"),
            json_data.get("image"),
            json_data.get("code")
        )
        if result:
            self.write(result)


class NextCaptchaHandler(tornado.web.RequestHandler):

    def get(self):
        """
        output:
        >>> {
        >>>     "status": "-1 / 0 / 1",
        >>>     "msg": "string"
        >>> }
        """
        ml = self.get_argument("model", None)
        if ml:
            return self.write(model.nextCaptcha(ml))
        self.write({
            "status": -1,
            "msg": "未指定model",
            "data": None
        })


def run_app():
    app = tornado.web.Application([
        ('/captcha', CaptchaHandler),
        # ('/captcha/save', SaveCaptchaHandler),  # todo, 是否需要在线标注功能呢?
        # ('/captcha/next', NextCaptchaHandler)
    ])
    app.listen(port=options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    run_app()
