# !/usr/bin/python
# -*- coding: UTF-8 -*-

import logging
import os
import sys
import time
import logging.handlers
from flask.logging import default_handler
from config1.config import Config
from config1.config_base import ConfigBase

from app import app
from app.common.c_eureka import eureka_register
from app.common.apollo_client import ApolloClient

def init_log():
    # 创建一个handler，用于写入日志文件
    log_path = str()
    if Config.Curr_EureakaEnv is ConfigBase.EureakaEnv.dev:
        log_path = os.path.join(ConfigBase.CONFIG_FILE_DIR_PATH, 'logs/')
    else:
        log_path = '/logs/'

    if os.path.exists(log_path) is False:
        os.mkdir(log_path)
    log_name = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    log_name += '.log'
    log_name = log_path + log_name
    logging_level = logging.DEBUG
    ch = logging.StreamHandler(stream=sys.stdout)
    fh = logging.handlers.RotatingFileHandler(filename=log_name, encoding='utf-8', maxBytes=1024 * 1024 * 10,
                                              backupCount=10000)
    ch.setLevel(logging_level)
    fh.setLevel(logging_level)

    # 定义handler的输出格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(process)d - %(thread)d - %(module)s - %(funcName)s - %(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    app.logger.addHandler(ch)
    # app.logger.addHandler(fh)
    app.logger.setLevel(logging_level)
    app.logger.removeHandler(default_handler)


def init_server():
    init_log()
    app.logger.info('Start init the server')
    if not os.path.exists(ConfigBase.CONFIG_FILE_DIR_PATH):
        os.mkdir(ConfigBase.CONFIG_FILE_DIR_PATH)
    # todo
    eureka_register(ConfigBase.PORT)
    client = ApolloClient(config_server_url=Config.ApolloConfig.APOLLO_HOST, app_id=Config.ApolloConfig.APOLLO_APP,
                          timeout=20)
    client.start(catch_signals=False)
    app.logger.info('Finish init the server')
    pass


init_server()




