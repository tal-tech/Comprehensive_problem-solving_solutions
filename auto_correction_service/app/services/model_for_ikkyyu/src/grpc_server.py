#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import grpc
import time
import cv2
import base64
import json
import numpy as np
import psutil
import functools
import sys
import threading
import ctypes
import inspect
import time
from logger import logger
from concurrent import futures
from rpc import facethink_pb2_grpc, facethink_pb2
from pipeline import pipeline

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '0.0.0.0'
_PORT = '8008'
TIMEOUT = 9
THREAD_NUM = 2


# 终止线程
def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        return 'invalid thread id'
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        return 'PyThreadState_SetAsyncExc failed'


class TimeOutException(Exception):
    pass


class Ikkyyu(facethink_pb2_grpc.grpcServerServicer):
    def __init__(self):
        self.pipe = pipeline()

    def process(self, request, context):
        logger.info('peer:{}, size:{}, 内存占用率：{}%'.format(context.peer(), request.ByteSize(), 
                                                         str(psutil.virtual_memory().percent)))
        try:
            img = json.loads(request.data_json)['img']
            img = base64.b64decode(img)
            buf = np.asarray(bytearray(img), dtype="uint8")
            tmp = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            task = threading.Thread(target=self.pipe.grpc, args=(tmp, ))
            task.start()
            task.join(TIMEOUT)
            if task.isAlive():
                stop_thread(task)
                raise TimeOutException("Processing Timeout")
            dic = {'code': 200, 'ret': self.pipe.ret}
            res = json.dumps(dic, ensure_ascii=False)
        except KeyError:
            logger.error("请求JSON串里必须有img字段，值为图片的base64编码后的值")
            res = '{"code": 400, "ret": "请求JSON串里必须有img字段，值为图片的base64编码后的值"}'
        except TimeOutException:
            res = '{"code": 12}'
        except Exception as e:
            res = '{"code": 500, "ret": "%s"}' % str(e)
            logger.error(res)
        return facethink_pb2.ModelProto(data_json=res)


def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=THREAD_NUM), options=[
          ('grpc.max_send_message_length', 50 * 1024 * 1024),
          ('grpc.max_receive_message_length', 50 * 1024 * 1024)
      ])
    facethink_pb2_grpc.add_grpcServerServicer_to_server(Ikkyyu(), grpcServer)
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)
    grpcServer.start()
    logger.info('Service started!')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)


if __name__ == '__main__':
    serve()
