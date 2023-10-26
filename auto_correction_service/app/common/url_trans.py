# !/usr/bin/python
# -*- coding: UTF-8 -*-
#  外部url转录为内部oss url
import requests
# import json
# import uuid
import time
from app import app
from config1.config import Config
from app.common.tool_unit import func_time

@func_time
def url_trans_process(requestid, url):
    info = {'urls': [url],
            'requestId': requestid,
            'sendTime': int(time.time() * 1000)}
    inner_url = None
    inner_url_len = 0
    try_count = 1
    while try_count <= 1:
        try:
            app.logger.info(
                'Task id is:{}, call url trans server {} times'.format(requestid, try_count))
            res = requests.post(Config.DATA_FLOW_URL_TRANS, json=info, headers={'Connection': 'close'}, timeout=1)
            if requests.codes.ok == res.status_code:
                res_json = res.json()
                if res_json['code'] == 2000000:
                    trans_url_list = res_json['resultBean']
                    for trans_url in trans_url_list:
                        inner_url = trans_url['innerUrl']
                        inner_url_len = trans_url['length']
                    app.logger.info('Task id is:{}, url trans sucess, oss url:{}'.format(requestid,
                                                                                                 url))
                    break
                else:
                    try_count += 1
                    app.logger.info('Task id is:{}, url trans failed'.format(requestid))
            else:
                try_count += 1
                app.logger.error(
                    'alertcode:91001009, altermsg:Task id {}, call trans server error'.format(requestid))
        except Exception as e:
            try_count += 1
            app.logger.error(
                'alertcode:91001009, altermsg:Task id is {}, call trans server:{}'.format(requestid,
                                                                                          str(e)))
    return inner_url, inner_url_len
# if __name__ == '__main__':
#     print('run')
#
#     file_path = '/Users/tal/work/project/ai_platform/ges/souti/search_questions_service/demo/8.jpeg'
#     with open(file_path, 'rb') as fp:
#         data_bin = fp.read()
#     ret = http_send(1, data_bin, '1+1')
#     print(ret)
