def catchErrorAndRetunDefault(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {
                "status": -1,
                "msg": f"后台运行错误: {e.__repr__()}",
                "data": None
            }

    return wrapper
