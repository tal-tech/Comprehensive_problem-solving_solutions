import os
import cv2
import json
import tensorflow as tf
slim = tf.contrib.slim
import cfg
from nets import all as ssdnet
from nets import np_methods
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
            area = np_methods.bboxes_area(bboxes[i])
            if area > 0.3 or area < 0:
                continue
            elif classes[i] ==1 and area > 0.2:
                continue
            # top, left, bottom, right
            ssd_rs.append([cls_id, score, bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]])
    return ssd_rs

def get_rect(bbox):
    x1 = float(bbox[0]) # w
    y1 = float(bbox[1]) # h
    x2 = float(bbox[2])
    y2 = float(bbox[3])
    x3 = float(bbox[4])
    y3 = float(bbox[5])
    x4 = float(bbox[6])
    y4 = float(bbox[7])
    xmin = min([x1, x2, x3, x4])
    xmax = max([x1, x2, x3, x4])
    ymin = min([y1, y2, y3, y4])
    ymax = max([y1, y2, y3, y4])
    return ymin, xmin, ymax, xmax


def get_post_result(ssd_rs, east_rs, shape):
    rs = []
    #ssd_rs: result from ssd, [label, top, left, bottom, right]
    text_bboxes = []
    east_num = len(east_rs)
    flag = [False] * east_num
    for ssd_bbox in ssd_rs:
        if ssd_bbox[0]!=1:
            d = {}
            d['label'] = ssd_bbox[0]
            d['parent'] = cvtCoor(ssd_bbox[2:], shape)
            d['children'] = []
            rs.append(d)
        else:
            ssd_bbox = [int(ssd_bbox[2] * shape[0]), int(ssd_bbox[3] * shape[1]), int(ssd_bbox[4] * shape[0]), int(ssd_bbox[5] * shape[1])]
            # h w h w
            for index, east_bbox in enumerate(east_rs):
                if not flag[index]:
                    east_bbox_rec = get_rect(east_bbox)
                    iou = IOS_calculation(ssd_bbox, east_bbox_rec)
                    if iou > 0.65:
                        flag[index] = True
                        text_bboxes.append(east_bbox)
                        break

                if index == (east_num - 1):
                    text_bboxes.append(ssd_bbox)
    for index, east_bbox in enumerate(east_rs):
        if not flag[index]:
            text_bboxes.append(east_bbox)
    for text_bbox in text_bboxes:
        if len(text_bbox) == 8:
            text_bbox_rect = get_rect(text_bbox)
        else:
            text_bbox_rect = text_bbox
            text_bbox = [text_bbox[1], text_bbox[0], text_bbox[3], text_bbox[2]] #w h w h
        h_box = text_bbox_rect[2] - text_bbox_rect[0]
        # w_box = text_bbox_rect[3] - text_bbox_rect[1]
        h_img = shape[0]
        if h_box / h_img < 0.015:
            continue
        rs = add_into_rs(text_bbox, text_bbox_rect, rs, shape)
    # rs: [left, top, right, bottom]
    for parentBox in rs:
        if parentBox['label'] != 1:
            parentBox['children'].sort(key=lambda x: x[1])

    return rs


def include(hs_box, big_box):
    hs_center_x = (hs_box[0] + hs_box[2]) / 2
    hs_center_y = (hs_box[1] + hs_box[3]) / 2
    w = big_box[2] - big_box[0]
    h = big_box[3] - big_box[1]
    fac = 0.2
    if hs_center_x > (big_box[0] + w * fac) and hs_center_x < (big_box[2] - w * fac) and \
            hs_center_y > (big_box[1] + h * fac) and hs_center_y < (big_box[3] - h * fac):
        return True
    else:
        return False


def add_into_rs(bbox, bbox_rect, rs, shape):
    """
    将大框里的小横式跟大框对应上
    """
    # coord = {}
    # coord["eight"] = bbox
    # coord["four"] = bbox_rect
    hs_box = [bbox_rect[1], bbox_rect[0], bbox_rect[3], bbox_rect[2]]
    # w h w h
    flag = True
    for j, ssd_bbox in enumerate(rs):
        if ssd_bbox['label'] == 2 or ssd_bbox['label'] == 3:
            ios = IOS_calculation(hs_box, ssd_bbox['parent'])
            if ios > cfg.IOSthreshold:
                ssd_bbox['children'].append(bbox)
                rs[j] = ssd_bbox
                if ios > cfg.IOSthreshold_high or ssd_bbox['label'] == 3 or include(hs_box, ssd_bbox['parent']):
                    flag = False
    if flag:
        d = {}
        d['label'] = 1
        d['parent'] = bbox
        d['children'] = []
        rs.append(d)
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



def process(obj_ssd, obj_east, img, shape):
    classes, scores, bboxes = obj_ssd.inference(img)
    east_rs = obj_east.east_process(img)
    ssd_rs = get_det_result(classes, scores, bboxes)
    if cfg.MULTIPROCESS:
        ssd_rs = postprocess_via_ratio(obj_ssd, img, ssd_rs, shape)
    rs = get_post_result(ssd_rs, east_rs, shape)
    return rs
    
    
def visulization(rs, img, result_path, name):
    c = cfg.COLORS[4]
    for ssd_bbox in rs:
        bbox = ssd_bbox['parent']
        if ssd_bbox['label'] != 1:

            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), cfg.COLORS[ssd_bbox['label']],2)
        else:
            cv2.line(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), c, 2)
            cv2.line(img, (int(bbox[2]), int(bbox[3])), (int(bbox[4]), int(bbox[5])), c, 2)
            cv2.line(img, (int(bbox[4]), int(bbox[5])), (int(bbox[6]), int(bbox[7])), c, 2)
            cv2.line(img, (int(bbox[6]), int(bbox[7])), (int(bbox[0]), int(bbox[1])), c, 2)
        for bbox in ssd_bbox['children']:
            cv2.line(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), c, 2)
            cv2.line(img, (int(bbox[2]), int(bbox[3])), (int(bbox[4]), int(bbox[5])), c, 2)
            cv2.line(img, (int(bbox[4]), int(bbox[5])), (int(bbox[6]), int(bbox[7])), c, 2)
            cv2.line(img, (int(bbox[6]), int(bbox[7])), (int(bbox[0]), int(bbox[1])), c, 2)
            # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), cfg.COLORS[4],2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_name = 'demo_%s'%(name)
    cv2.imwrite(os.path.join(result_path, img_name), img)


if __name__ == "__main__":
    gpu_id = '0'
    import sys
    sys.path.append('./')
    sys.path.append('../')
    sys.path.append('../east_tf/')
    from east_tf.east_inference import East
    obj_east = East('/home/kk/pipeline_for_ikkyyu/Models/east/model.ckpt-V0.6.19')
    obj_ssd = det_Layout(gpu_id, '/home/kk/pipeline_for_ikkyyu/Models/ssd/model.ckpt-V0.6.12')
    # input image
    result_path = './'
    image_names = '/home/kk/pipeline_for_ikkyyu/badcase/27379img_20190210_003106.jpg'
#     image_names[index] = '1.jpg'
    try:
        img = cv2.imread(image_names)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print("Image does not exist or is malformed. Please check the input image.")
    shape = img.shape
     # inference
    rs = process(obj_ssd, obj_east, img, shape)
    # save json
    json_name = 'infer.json'
    with open(os.path.join(result_path, json_name), 'w') as f:
        f.write(json.dumps(rs, indent=4, ensure_ascii=False))
        f.close()
    # visualization
    visulization(rs, img, result_path, image_names.split('/')[-1], shape)
