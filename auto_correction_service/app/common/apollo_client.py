# -*- coding: utf-8 -*-
import json
import sys
import threading
import time
import requests
from app import app
from config1.config import Config


class ApolloClient(object):
    def __init__(self, config_server_url, app_id, cluster='default', timeout=35, ip=None):
        self.config_server_url = config_server_url
        self.appId = app_id
        self.cluster = cluster
        self.timeout = timeout
        self.stopped = False
        self.init_ip(ip)

        self._stopping = False
        self._cache = {}
        self._notification_map = {'datawork-common': -1}
        app.logger.info("ApolloClient init finish,url:{},app_id:{},ip:{}".format(config_server_url, app_id, ip))


    def init_ip(self, ip):
        if ip:
            self.ip = ip
        else:
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('8.8.8.8', 53))
                ip = s.getsockname()[0]
            finally:
                s.close()
            self.ip = ip

    # Main method
    def get_value(self, key, default_val=None, namespace='application', auto_fetch_on_cache_miss=False):
        if namespace not in self._notification_map:
            self._notification_map[namespace] = -1
            app.logger.info("Add namespace '%s' to local notification map", namespace)

        if namespace not in self._cache:
            self._cache[namespace] = {}
            app.logger.info("Add namespace '%s' to local cache", namespace)
            # This is a new namespace, need to do a blocking fetch to populate the local cache
            self._long_poll()

        if key in self._cache[namespace]:
            return self._cache[namespace][key]
        else:
            if auto_fetch_on_cache_miss:
                return self._cached_http_get(key, default_val, namespace)
            else:
                return default_val
    
    def update_config(self, config_dict):
        try:
            app.logger.info('apollo server update config, data:{}'.format(config_dict))
            for key in config_dict:
                data = config_dict[key]
                if 'image' == key:
                    Config.ApolloConfig.DataFlow.DF_topic = data
                if 'kafka-bootstrap-servers' == key:
                    Config.ApolloConfig.DataFlow.DF_MQ_HOST = data

        except Exception as e:
            pass

    # Start the long polling loop. Two modes are provided:
    # 1: thread mode (default), create a worker thread to do the loop. Call self.stop() to quit the loop
    # 2: eventlet mode (recommended), no need to call the .stop() since it is async
    def start(self, use_eventlet=False, eventlet_monkey_patch=False, catch_signals=True):
        # First do a blocking long poll to populate the local cache, otherwise we may get racing problems
        if len(self._cache) == 0:
            self._long_poll()
        if use_eventlet:
            import eventlet
            if eventlet_monkey_patch:
                eventlet.monkey_patch()
            eventlet.spawn(self._listener)
        else:
            if catch_signals:
                import signal
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                signal.signal(signal.SIGABRT, self._signal_handler)
            t = threading.Thread(target=self._listener)
            t.start()

    def stop(self):
        self._stopping = True
        app.logger.info("Stopping listener...")

    def _cached_http_get(self, key, default_val, namespace='application'):
        url = '{}/configfiles/json/{}/{}/{}?ip={}'.format(self.config_server_url, self.appId, self.cluster, namespace, self.ip)
        r = requests.get(url)
        print(url)
        if r.ok:
            data = r.json()
            self._cache[namespace] = data
            app.logger.info('Updated local cache for namespace %s', namespace)
        else:
            data = self._cache[namespace]

        if key in data:
            return data[key]
        else:
            return default_val

    def _uncached_http_get(self, namespace='application'):
        url = '{}/configs/{}/{}/{}?ip={}'.format(self.config_server_url, self.appId, self.cluster, namespace, self.ip)
        r = requests.get(url)
        print(url)
        if r.status_code == 200:
            data = r.json()
            self._cache[namespace] = data['configurations']
            self.update_config(self._cache[namespace])
            app.logger.info('Updated local cache for namespace %s release key %s: %s',
                                             namespace, data['releaseKey'],
                                             repr(self._cache[namespace]))

    def _signal_handler(self, signal, frame):
        app.logger.info('You pressed Ctrl+C!')
        self._stopping = True

    def _long_poll(self):
        try:
            url = '{}/notifications/v2'.format(self.config_server_url)
            notifications = []
            for key in self._notification_map:
                notification_id = self._notification_map[key]
                notifications.append({
                    'namespaceName': key,
                    'notificationId': notification_id
                })

            r = requests.get(url=url, params={
                'appId': self.appId,
                'cluster': self.cluster,
                'notifications': json.dumps(notifications, ensure_ascii=False)
            }, timeout=self.timeout)

            app.logger.debug('Long polling returns %d: url=%s', r.status_code, r.request.url)

            if r.status_code == 304:
                # no change, loop
                app.logger.debug('No change, loop...')
                return

            if r.status_code == 200:
                data = r.json()
                for entry in data:
                    ns = entry['namespaceName']
                    nid = entry['notificationId']
                    app.logger.info("%s has changes: notificationId=%d", ns, nid)
                    self._uncached_http_get(ns)
                    self._notification_map[ns] = nid
            else:
                app.logger.warn('Sleep...')
                time.sleep(self.timeout)
        except Exception as e:
            pass

    def _listener(self):
        app.logger.info('Entering listener loop...')
        while not self._stopping:
            self._long_poll()

        app.logger.info("Listener stopped!")
        self.stopped = True
