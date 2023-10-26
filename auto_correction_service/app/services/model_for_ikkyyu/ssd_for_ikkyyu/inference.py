import os
import time
import cv2
import json
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import cfg
from nets import all as ssdnet
from nets import ssd_common, np_methods
from utils.util import IOU_calculation, IOS_calculation
from preprocessing import s3d_preprocessing as preprocessing
class det_Layout(object):
    def __init__(self, deviceid, CKPT_PATH):
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(deviceid)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = cfg.GPU_FRACTION)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.img_input = tf.placeholder(tf.float32, shape = (None, None, 3))
        img_input, glabels, gbboxes, self.bbox_img = preprocessing.preprocess_for_eval(self.img_input, 
                   None, None, cfg.NET_SHAPE, cfg.DATA_FORMAT)
        img_input_ext = tf.expand_dims(img_input, 0)

        # net
        ssd_params = ssdnet.SSDNet.default_params._replace(num_classes=cfg.CLASS_NUM)
        ssd = ssdnet.SSDNet(ssd_params)

        self.ssd_anchors = ssd.anchors(ssd_params.img_shape)
        arg_scope = ssd.arg_scope(data_format = cfg.DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            self.predictions, self.localisations, _, _ = ssd.net(img_input_ext, is_training=False)

        self.isess=tf.Session(config = config)
        saver = tf.train.Saver()
        saver.restore(self.isess, CKPT_PATH)

    def process_image(self, img):
        rpredictions, rlocalisations, rbbox_img = self.isess.run([self.predictions, 
                                        self.localisations, self.bbox_img], feed_dict = {self.img_input: img})
        return rpredictions, rlocalisations, rbbox_img
    
    def postprocess(self, rpredictions, rlocalisations, rbbox_img, select_threshold = cfg.SELECT_THRESHOLD, nms_threshold = cfg.NMS_THRESHOLD):
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(rpredictions, rlocalisations, self.ssd_anchors, 
                select_threshold = select_threshold, img_shape = cfg.NET_SHAPE, num_classes = cfg.CLASS_NUM, decode = True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)           
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k = cfg.TOP_K)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold = nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes
    
    def inference(self, img, select_threshold = cfg.SELECT_THRESHOLD, nms_threshold = cfg.NMS_THRESHOLD):
        rpredictions, rlocalisations, rbbox_img = self.process_image(img)
        rclasses, rscores, rbboxes = self.postprocess(rpredictions, rlocalisations, rbbox_img, 
                                                      select_threshold = select_threshold, nms_threshold = nms_threshold)
        return rclasses, rscores, rbboxes
    
    
def cvtCoor(bbox, shape):
    """
    param:
        bbox: [top, left, bottom, right] 相对坐标
        shape: image shape
    return:
        rs: [left, top, right, bottom] 绝对坐标
    """
    height = shape[0]
    width = shape[1]
    return [int(bbox[1] * width), int(bbox[0] * height), int(bbox[3] * width), int(bbox[2] * height)]
    
    
def add_into_rs(bbox, rs, shape):
    """
    将大框里的小横式跟大框对应上
    """
    height = shape[0]
    width = shape[1]
    hs_box = [bbox[1] * width, bbox[0] * height, bbox[3] * width , bbox[2] * height]
    flag = True
    for j, ssd_bbox in enumerate(rs):
        if ssd_bbox['label'] == 2 or ssd_bbox['label'] == 3:
            ios = IOS_calculation(hs_box, ssd_bbox['parent'])
            if ios > cfg.IOSthreshold:
                ssd_bbox['children'].append(cvtCoor(bbox, shape))
                rs[j] = ssd_bbox
                if ios > cfg.IOSthreshold_high or ssd_bbox['label'] == 3:
                    flag = False
    if flag:
        d = {}
        d['label'] = 1
        d['parent'] = cvtCoor(bbox, shape)
        d['children'] = []
        rs.append(d)
    return rs
    

def get_det_result(classes, scores, bboxes):
    """
    param:
        shape: image shape
    return:
        ssd_rs: result from ssd, [label, top, left, bottom, right]
    """
    # post process
    ssd_rs = []
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id > 0:
            score = scores[i] 
            # top, left, bottom, right
            ssd_rs.append([cls_id, score, bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]])
    return ssd_rs


def get_post_result(ssd_rs, shape):
    rs = []
    #ssd_rs: result from ssd, [label, top, left, bottom, right]
    for ssd_bbox in ssd_rs:
        if ssd_bbox[0]!=1:
            d = {}
            d['label'] = ssd_bbox[0]
            d['parent'] = cvtCoor(ssd_bbox[2:], shape)
            d['children'] = []
            rs.append(d)
    for ssd_bbox in ssd_rs:
        if ssd_bbox[0] == 1:
            rs = add_into_rs(ssd_bbox[2:], rs, shape)
    # rs: [left, top, right, bottom]
    for parentBox in rs:
        if parentBox['label'] != 1:
            parentBox['children'].sort(key=lambda x: x[1] + x[3])
    return rs



def postprocess_via_ratio(obj, img, ssd_rs, shape, select_threshold = cfg.SELECT_THRESHOLD, nms_threshold = cfg.NMS_THRESHOLD):
    ratio = shape[0] / shape[1]          
    if ratio > 1.25:
        img_up = img[:int(0.6 * shape[0]),:,:]
        classes_up, scores_up, bboxes_up = obj.inference(img_up, select_threshold = select_threshold, nms_threshold = nms_threshold)
        ssd_rs_up = get_det_result(classes_up, scores_up, bboxes_up)
        
        img_down = img[int(0.4 * shape[0]):,:,:]
        classes_down, scores_down, bboxes_down =  obj.inference(img_down, select_threshold = select_threshold, nms_threshold = nms_threshold)
        ssd_rs_down = get_det_result(classes_down, scores_down, bboxes_down)
        
        #上半图
        for ssd_bbox_up in ssd_rs_up: 
            bbox_up = [ssd_bbox_up[0], ssd_bbox_up[1], ssd_bbox_up[2] * 0.6, ssd_bbox_up[3], ssd_bbox_up[4] * 0.6, ssd_bbox_up[5]]
            flag = False
            for ssd_bbox in ssd_rs:
                if ssd_bbox_up[0] == ssd_bbox[0]:
                    if IOU_calculation(bbox_up[2:], ssd_bbox[2:]) > cfg.IOUthreshold:
                        flag = True
                        break
            if not flag:
                ssd_rs.append(bbox_up)
        #下半图      
        for ssd_bbox_down in ssd_rs_down: 
            bbox_down = [ssd_bbox_down[0], ssd_bbox_down[1], ssd_bbox_down[2] * 0.6 + 0.4, ssd_bbox_down[3], ssd_bbox_down[4] * 0.6 + 0.4, ssd_bbox_down[5]]
            flag = False
            for ssd_bbox in ssd_rs:
                if ssd_bbox_down[0] == ssd_bbox[0]:
                    if IOU_calculation(bbox_down[2:], ssd_bbox[2:]) > cfg.IOUthreshold:
                        flag = True
                        break
            if not flag:
                ssd_rs.append(bbox_down)
        
    elif ratio < 0.8:
        img_left = img[:,:int(0.6 * shape[1]),:]
        classes_left, scores_left, bboxes_left = obj.inference(img_left, select_threshold = select_threshold, nms_threshold = nms_threshold)
        ssd_rs_left = get_det_result(classes_left, scores_left, bboxes_left)
        
        img_right = img[:,int(0.4 * shape[1]):,:]
        classes_right, scores_right, bboxes_right = obj.inference(img_right, select_threshold = select_threshold, nms_threshold = nms_threshold)
        ssd_rs_right = get_det_result(classes_right, scores_right, bboxes_right)
        
        #左半图
        for ssd_bbox_left in ssd_rs_left: 
            bbox_left = [ssd_bbox_left[0], ssd_bbox_left[1], ssd_bbox_left[2], ssd_bbox_left[3] * 0.6, ssd_bbox_left[4], ssd_bbox_left[5] * 0.6]
            flag = False
            for ssd_bbox in ssd_rs:
                if ssd_bbox_left[0] == ssd_bbox[0]:
                    if IOU_calculation(bbox_left[2:], ssd_bbox[2:]) > cfg.IOUthreshold:
                        flag = True
                        break
            if not flag:
                ssd_rs.append(bbox_left)
        #右半图      
        for ssd_bbox_right in ssd_rs_right: 
            bbox_right = [ssd_bbox_right[0], ssd_bbox_right[1], ssd_bbox_right[2], ssd_bbox_right[3] * 0.6 + 0.4, ssd_bbox_right[4], ssd_bbox_right[5] * 0.6 + 0.4]
            flag = False
            for ssd_bbox in ssd_rs:
                if ssd_bbox_right[0] == ssd_bbox[0]:
                    if IOU_calculation(bbox_right[2:], ssd_bbox[2:]) > cfg.IOUthreshold:
                        flag = True
                        break
            if not flag:
                ssd_rs.append(bbox_right)                      
    return ssd_rs    



def process(obj, img, shape):
    classes, scores, bboxes = obj.inference(img)

    ssd_rs = get_det_result(classes, scores, bboxes)
    if cfg.MULTIPROCESS:
        ssd_rs = postprocess_via_ratio(obj, img, ssd_rs, shape)
    rs = get_post_result(ssd_rs, shape)
    return rs
    
    
def visulization(rs, img, result_path, name, shape):
    height = shape[0]
    width =  shape[1]
    for ssd_bbox in rs:
        bbox = ssd_bbox['parent']    
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), cfg.COLORS[ssd_bbox['label']],2)
        for bbox in ssd_bbox['children']:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), cfg.COLORS[4],2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_name = 'demo_%s'%(name)
    cv2.imwrite(os.path.join(result_path, img_name), img)


if __name__ == "__main__":
    gpu_id = '0'
    obj = det_Layout(gpu_id, '/home/kk/pipeline_for_ikkyyu/Models/ssd/model.ckpt-V0.6.12')
    # input image
    img_path = './'
    result_path = './'
    image_names = '/home/kk/pipeline_for_ikkyyu/badcase/27379img_20190210_003106.jpg'
    try:
        img = cv2.imread(image_names)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print("Image does not exist or is malformed. Please check the input image.")
    shape = img.shape
     # inference
    rs = process(obj, img, shape)
    # save json
    json_name = 'infer.json'
    with open(os.path.join(result_path, json_name), 'w') as f:
        f.write(json.dumps(rs, indent=4, ensure_ascii=False))
        f.close()
    # visualization
    visulization(rs, img, result_path, image_names.split('/')[-1], shape)
