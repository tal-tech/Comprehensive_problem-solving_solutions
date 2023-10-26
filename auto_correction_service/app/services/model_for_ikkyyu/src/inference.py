import os
import cfg
import time
import cv2
import json
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from nets import pixel_anchor2 as ssdnet
from nets import ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing


class det_Layout(object):
    def __init__(self, deviceid):
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(deviceid)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = cfg.GPU_FRACTION)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.img_input = tf.placeholder(tf.float32, shape = (None, None, 3))
        img_input, glabels, gbboxes, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval
                (self.img_input, None, None, cfg.NET_SHAPE, cfg.DATA_FORMAT, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
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
        saver.restore(self.isess, cfg.CKPT_PATH)



    def process_image(self, img, select_threshold = cfg.SELECT_THRESHOLD, nms_threshold = cfg.NMS_THRESHOLD):
        rpredictions, rlocalisations, rbbox_img = 
                self.isess.run([self.predictions, self.localisations, self.bbox_img], feed_dict = {self.img_input: img})
        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(rpredictions, rlocalisations, self.ssd_anchors, 
                select_threshold = select_threshold, img_shape = cfg.NET_SHAPE, num_classes = cfg.CLASS_NUM, decode = True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k = cfg.TOP_K)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold = nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes


def is_contain(coord1, coord2):
    if coord1[0] >= coord2[0] and coord1[1] >= coord2[1] and coord1[2] <= coord2[2] and coord1[3] <= coord2[3]:
        return True
    return False


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
    hs_box = [bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width]
    flag = True
    for j, ssd_bbox in enumerate(rs):
        if ssd_bbox['label'] == 2 or ssd_bbox['label'] == 3:
            big_box = [ssd_bbox['parent'][1] - 0.02 * height, ssd_bbox['parent'][0] - 0.02 * width, ssd_bbox['parent'][3] + 0.02 * height, ssd_bbox['parent'][2] + 0.02 * width] 
            if is_contain(hs_box, big_box):
                ssd_bbox['children'].append(cvtCoor(bbox, shape))
                rs[j] = ssd_bbox
                flag = False
    if flag:
        d = {}
        d['label'] = 1
        d['parent'] = cvtCoor(bbox, shape)
        d['children'] = []
        rs.append(d)
    return rs
    

def bbox_filter(ssd_rs, shape):
    """
    param:
        ssd_rs: result from ssd, [label, top, left, bottom, right]
        shape: image shape
    return:
        rs: result after filter
    """
    rs = []
    for ssd_bbox in ssd_rs:
        if ssd_bbox[0]!=1:
            d = {}
            d['label'] = ssd_bbox[0]
            d['parent'] = cvtCoor(ssd_bbox[1:], shape)
            d['children'] = []
            rs.append(d)
    for ssd_bbox in ssd_rs:
        if ssd_bbox[0] == 1:
            rs = add_into_rs(ssd_bbox[1:], rs, shape)
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
    obj = det_Layout(gpu_id)
    # input image
    img_path = './demo/imgs/'
    result_path = './demo/results/'
    image_names = os.listdir(img_path)
#     index = np.random.randint(len(image_names))
    index = 0
    print(image_names[index])
    try:
        img = cv2.imread(img_path + image_names[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print("Image does not exist or is malformed. Please check the input image.")
    shape = img.shape
    
    # inference
    classes, scores, bboxes =  obj.process_image(img)
    
    # post process
    ssd_rs = []
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id > 0:
            score = scores[i]
            # top, left, bottom, right
            ssd_rs.append([cls_id, bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]])
    rs = bbox_filter(ssd_rs, shape)
    
    # save json
    json_name = 'infer_%s.json'%(image_names[index][:-4])
    with open(os.path.join(result_path, json_name), 'w') as f:
        f.write(json.dumps(rs, indent=4, ensure_ascii=False))
        f.close()
        
    # visualization
    visulization(rs, img, result_path, image_names[index], shape)
