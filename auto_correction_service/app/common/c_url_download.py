#!/usr/bin/python
# -*- coding: UTF-8 -*-
from app import app
import os
import requests
from app.common.error_msg import *
from urllib.parse import urlparse
from config1.config_base import ConfigBase
from app.common.tool_unit import func_time


"""
def file_download(url, file_path):
    try:
        # 获取文件名
        res = (urlparse(url)).path
        file_name = res.split('/')[-1]
        if file_name is None:
            app.logger.error(StatusCode[8], extra={'status_code': 8})
            return 8, StatusCode[8]

        r = requests.get(url, stream=True, headers={'Connection': 'close'})
        if not r:
            app.logger.error(StatusCode[9], extra={'status_code': 9})
            return 9, StatusCode[9]

        content_length = int(r.headers['content-length'])
        if content_length >= 1024 * 1024 * 5:
            app.logger.error(StatusCode[4], extra={'status_code': 4})
            return 4, StatusCode[4]

        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, "ab") as code:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    code.write(chunk)
            code.flush()

        return 0, file_path
    except Exception as e:
        app.logger.error(e, extra={'status_code': 104})
        return 3, e
"""

@func_time
def file_download(url):
    """
    下载图片
    :param url:
    :return: True,content || False, Status.IMG_DOWNLOAD_ERROR
    """
    try:
        r = requests.get(url, stream=True, headers={'Connection': 'close'})
        if not r:
            return False, Status.IMG_DOWNLOAD_ERROR
        if 'content-length' in r.headers and int(r.headers['content-length']) > ConfigBase.IMG_MAX_SIZE:
            return False, Status.FILE_TOO_LARGE_ERROR

        return True, r.content
    except Exception as e:
        return False, Status.IMG_DOWNLOAD_ERROR


if __name__ == '__main__':
    url = 'http://221.122.128.3/txn-test/ikkyyu_test.jpg?AWSAccessKeyId=OFHRQRBF1IC07YDYIMFE&Expires=2488431455&Signature=Kk8LyuAq48OhgO2efasvI6Xbeqk%3D'
    # file_download(url, 'E:/PROJECT/ikkyyu/ikkyyu-web-v0.1/mylog')
    # print('hello world!')
    url = 'http://img.xixik.net/custom/section/country-flag/xixik-cdaca66ba3839767.png'
    r = requests.get(url)
    if not r:
        print("false")
    print(requests.get(url).content)