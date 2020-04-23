import re

BASE64_REGEX = re.compile("^data:image/.+?;base64,").sub


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


def show_dynamic_ratio(cur_count, all_count, text='rate'):
    ratio = cur_count / all_count
    dynamic_ratio = int(ratio * 50)
    dynamic = '#' * dynamic_ratio + ' ' * (50 - dynamic_ratio)
    percentage = int(ratio * 100)
    print("\r[{}] {}: {}/{} {}%".format(dynamic, text, cur_count, all_count, percentage),
          end='', flush=True)
