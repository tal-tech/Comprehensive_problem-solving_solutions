#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from enum import Enum, unique


@unique
class Status(Enum):
    SUCCESS = {20000: 'ok'}
    REQ_PARAMS_ERROR = {300018001: 'error input param'}
    IMG_DOWNLOAD_ERROR = {300018002: 'image file download failed'}
    FILE_TOO_LARGE_ERROR = {300018003: 'The picture file is more than 5M'}
    IMG_TYPE_ERROR = {300018004: 'Unsupported image file types'}
    PIC_INFO_ERROR = {300018005: 'Incorrect picture information'}
    FILE_NO_EXIST_ERROR = {300018006: 'File name does not exist'}
    URL_REQ_ERROR = {300018007: 'Url request failed'}
    MODEL_ANALYSIS_ERROR = {300018008: 'Model analysis failed'}
    # SEARCH_SERVICE_FAILED = {300018009: '搜索服务不可用'}
    # SEARCH_SERVICE_ERROR = {300018010: '搜索服务出错'}
    #
    # TOKEN_ERROR = {300018051: '调用token失败'}
    # QINGZHOU_SERVICE_ERROR = {300018052: '调用服务失败'}
    UNKNOWN_ERROR = {300018099: 'unknown error'}

    def err_code(self):
        return tuple(self.value.keys())[0]

    def err_msg(self):
        return tuple(self.value.values())[0]


class MyExcept(Exception):
    def __init__(self, err_info, db_id=None):
        self.err_code = err_info.err_code()
        self.err_msg = err_info.err_msg()
        self.db_id = db_id
        self.err_state = err_info

    def __str__(self):
        return self.err_info

    def err_status(self):
        return self.err_state


def code2stat_msg(code):
    for i, j in Status.__members__.items():
        if j.err_code() == code:
            return j
    return None
