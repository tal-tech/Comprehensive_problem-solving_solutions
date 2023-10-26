import json
import time
import threading
from pykafka import KafkaClient
from app import app
from config1.config import Config
from app.common.singleton import SingletonIns

'''
bizType: datawork-text, datawork-speech, datawork-image, datawork-other
'''


class MQBean(object):
    def __init__(self):
        self.__produce_data = dict()

    def __check_type(self):
        int_vals = ['requestTime', 'responseTime', 'code', 'errCode', 'duration', 'sendTime']
        dict_vals = ['sourceRemark', 'data', 'dataRemark', 'tag']
        vector_vals = ['model', 'scene', 'subject', 'sourceInfos']
        str_vals = ['apiId', 'apiName', 'apiType', 'bizType', 'appKey', 'version', 'url', 'requestId',
                    'msg', 'errMsg', 'id', 'sourceType', 'content', 'fileType', 'encoding', 'resolution']
        for key in self.__produce_data.keys():
            v = self.__produce_data[key]
            if key in int_vals:
                if not isinstance(v, int):
                    raise ValueError('key:%s must be int!' % key)
            elif key in dict_vals:
                if not isinstance(v, dict):
                    raise ValueError('key:%s must be dict!' % key)
            elif key in vector_vals:
                if not isinstance(v, list):
                    raise ValueError('key:%s must be list!' % key)

                if key == 'sourceInfos':
                    for data in v:
                        for keys in data:
                            if not isinstance(data[keys], str):
                                raise ValueError('sourceInfo, key:%s must be str!' % keys)
                else:
                    for data in v:
                        if not isinstance(data, str):
                            raise ValueError('key:%s context must be str!' % key)
            elif key in str_vals:
                if not isinstance(v, str):
                    raise ValueError('key:%s must be str!' % key)

    def __must_exist(self):
        keys = ['bizType', 'version', 'url', 'requestId', 'sourceInfos']
        infos = ['id', 'sourceType', 'content']
        for key in keys:
            if key not in self.__produce_data.keys():
                raise ValueError('key:%s must be exist' % key)

        datas = self.__produce_data['sourceInfos']
        for key in infos:
            if key not in datas[0].keys():
                raise ValueError('sourceInfos[%s] must be exist' % (key))

    def set_produce_data(self, k, v):
        self.__produce_data[k] = v
        return True

    def get_produce_data(self, k):
        if k not in self.__produce_data.keys():
            return None
        return self.__produce_data[k]

    def get_topic(self):
        return Config.ApolloConfig.DataFlow.DF_topic

    @staticmethod
    def serialize(bean):
        bean.set_produce_data('sendTime', int(time.time() * 1000))
        bean.__check_type()
        bean.__must_exist()
        return json.dumps(bean.__produce_data)

    @staticmethod
    def deserialize(s):
        data = json.loads(s)
        bean = MQBean()
        bean.__produce_data = data
        bean.__check_type()
        bean.__must_exist()
        return bean


@SingletonIns
class MQClient(object):
    def __init__(self):
        self.__client = KafkaClient(hosts=Config.ApolloConfig.DataFlow.DF_MQ_HOST)
        self.__producer = dict()

    def __get_produce(self, key):
        if key in self.__producer:
            return self.__producer[key]
        else:
            topic = self.__client.topics[key]
            p = None
            try:
                p = topic.get_producer(max_request_size=1024*1024*10)
                self.__producer[key] = p
            except Exception as e:
                app.logger.info('get producer error:{}'.format(str(e)))
            return p

    def produce(self, bean):
        info = MQBean.serialize(bean)
        app.logger.debug('mq send kafka data:{}'.format(str(info)))
        try:
            if len(info) >= (1 << 23):
                raise ValueError('msg more than 8M')
        except Exception as e:
            app.logger.error('mq produce msg more than 8M')
            return

        p = self.__get_produce(bean.get_topic())
        while p == None:
            time.sleep(1)
            p = self.__get_produce(bean.get_topic())

        try:
            p.produce(info.encode())
        except Exception as e:
            app.logger.error('mq produce error, the error msg is:%s' % str(e))
