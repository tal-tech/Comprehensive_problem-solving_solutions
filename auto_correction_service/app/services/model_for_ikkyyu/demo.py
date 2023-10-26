import sys
import cv2
import  yaml
import json
import numpy as np
import os

#local lib
sys.path.append('./src')
sys.path.append(os.path.dirname(__file__)+'/src')
from src import pipeline
# import pipeline

# ikkyyu_pipeline = pipeline()  # 实例化pipeline类，并初始化参数
#
#     # 测试使用
# ikkyyu_pipeline.visual = True
# ikkyyu_pipeline.clock = True
# ikkyyu_pipeline.write = True
if __name__ == "__main__":

    ikkyyu_pipeline = pipeline()  # 实例化pipeline类，并初始化参数

    # 测试使用
    ikkyyu_pipeline.visual = True
    ikkyyu_pipeline.clock = True
    ikkyyu_pipeline.write = True
    ikkyyu_pipeline.aiqa = True


    # 批量图片检测
    # img_path = '/workspace/ikkyyu_pipeline/pipeline_for_ikkyyu/test_img'
    # imgs = os.listdir(img_path)
    # sorted(imgs)
    # for img in imgs:
    #     image = cv2.imread(os.path.join(img_path, img))
    #     try:
    #         print(img)
    #         ikkyyu_pipeline.grpc(image, img.split('.')[0], batch_size = 64)
    #         print(ikkyyu_pipeline.ret)
    #     except Exception as e:
    #         print(e)
    #         continue


    # # 单个图片检测
    for iteration in range(1):
        img_path = '/home/guoweiye/workspace/auto_correction_service/app/services/model_for_ikkyyu/images/8.jpeg'
        image = cv2.imread(img_path)
        ikkyyu_pipeline.grpc(image, 'test', batch_size = 64)
        print(ikkyyu_pipeline.ret)
