import uuid
import threading
import queue
from app import app
from app.common.c_eureka import app_name, SERVER_HOST
from app.common.mq_client import MQBean, MQClient


def call_mq_server(audio_task, result):
    mq_bean = MQBean()
    mq_bean.set_produce_data('apiId', audio_task.api_id)
    mq_bean.set_produce_data('apiName', app_name)
    mq_bean.set_produce_data('apiType', '0') # 0:同步  1:异步
    mq_bean.set_produce_data('bizType', 'datawork-image')
    mq_bean.set_produce_data('appKey', str(audio_task.app_key))
    mq_bean.set_produce_data('version', str(SERVER_HOST))
    mq_bean.set_produce_data('url', audio_task.req_url)
    mq_bean.set_produce_data('requestId', audio_task.request_id)
    mq_bean.set_produce_data('requestTime', audio_task.request_time)
    mq_bean.set_produce_data('responseTime', audio_task.response_time)
    mq_bean.set_produce_data('code', audio_task.status_paas_code)
    mq_bean.set_produce_data('msg', audio_task.code_msg)
    mq_bean.set_produce_data('errCode', audio_task.status_paas_code)
    mq_bean.set_produce_data('errMsg', audio_task.code_msg)
    mq_bean.set_produce_data('duration', int(audio_task.media_duration))

    source_info_dict = dict()
    source_info_dict['id'] = str(uuid.uuid1())
    source_info_dict['sourceType'] = 'base64'
    source_info_dict['content'] = str(audio_task.image_data)
    source_info_list = list()
    source_info_list.append(source_info_dict)
    mq_bean.set_produce_data('sourceInfos', source_info_list)

    source_dict = dict()
    source_dict['requestParam'] = str(audio_task.request_data)
    mq_bean.set_produce_data('sourceRemark', source_dict)

    mq_bean.set_produce_data('data', result)
    mq_bean.set_produce_data('dataRemark', dict())
    mq_bean.set_produce_data('tag', dict())

    mq_client = MQClient()
    mq_client.produce(mq_bean)
    app.logger.info('Task id:{}, mq send msg success'.format(audio_task.request_id))


class MQProcess(object):
    def __init__(self):
        self._close_thread = False
        self._mq_send_thread = None
        self._mq_send_queue = queue.Queue()
        self._mq_send_event = threading.Event()
        self.init()

    def init(self):
        self._mq_send_thread = threading.Thread(target=self.mq_process,
                                            args=(self._mq_send_queue, ))
        self._mq_send_thread.start()

    def dispatch_task(self, task, result):
        app.logger.info('Task id:{}, insert into mq_send_queue'.format(task.request_id))
        self._mq_send_queue.put_nowait([task, result])
        if self._mq_send_event and self._mq_send_event.is_set() is not True:
            self._mq_send_event.set()
            app.logger.info('Task id:{}, notify mq send thread!'.format(task.request_id))

    def mq_process(self, send_queue):
        while self._close_thread is False:
            if not send_queue.empty():
                try:
                    app.logger.info('pre deal queue size is [%d]' % send_queue.qsize())
                    info = send_queue.get()
                    if info is not None:
                        task = info[0]
                        result = info[1]
                        call_mq_server(task, result)
                except Exception as e:
                    app.logger.error('mq process error!, the error msg is {}'.format(str(e)))
            else:
                app.logger.info('mq process queue is empty, sleep')
                self._mq_send_event.clear()
                self._mq_send_event.wait()
