#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import time
from functools import wraps
from app import app


def func_time(f):
    """
    简单记录执行时间
    :param f:
    :return:
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        app.logger.debug('{} elapse time:{} ms'.format(f.__name__, int((end - start)*1000)))
        return result

    return wrapper
