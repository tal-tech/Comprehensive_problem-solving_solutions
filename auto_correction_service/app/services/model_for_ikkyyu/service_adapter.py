import sys
import os
import numpy as np
import cv2
from app.common.tool_unit import func_time
import json

sys.path.append(os.path.dirname(__file__) + '/src')
from pipeline import pipeline
from threading import Lock

lock = Lock()
ikkyyu_pipeline = pipeline()  # 实例化pipeline类，并初始化参数

# 测试使用
# ikkyyu_pipeline.visual = True
# ikkyyu_pipeline.clock = True
# ikkyyu_pipeline.write = True
# ikkyyu_pipeline.aiqa = True
sys.path.append('./src')

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

@func_time
def algorithm_process(img):
    """
    :param img:
    :return:
    # """
    try:
         # 转换为np数组
        img_array = np.fromstring(img, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # global lock
        # with lock:
        ikkyyu_pipeline.grpc(image, 'test', batch_size=64)
        #print(ikkyyu_pipeline.ret)
        # 转换为np数组
        res = {'code': 200, 'ret': ikkyyu_pipeline.ret}
        # TypeError: Object of type int32 is not JSON serializable  防止出现，进行预防
        res = json.dumps(res, ensure_ascii=False, cls = MyEncoder)
        res = json.loads(res)
    except Exception as e:
        res = {"code": 500, "ret": str(e)}
    return res

    # """
    # 1.检测返回值，检测 所需字段是否存在
    # 2. 返回 True:s False:s
    # 3.app.log.debug() 输出
    # :return:
    # """
    # ret = {'code': 200, 'ret': ''}
    #
    # return ret
