# -*- conding:utf-8 -*-
# encoding: utf-8
import json
import copy
import tensorflow as tf
import time
import cv2
import os
import cfg
import sys
import numpy as np
from tqdm import tqdm
from nets import np_methods
from inference import det_Layout, get_det_result, postprocess_via_ratio

ground_truth_dict={}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# cfg_ckpt_path = '/workspace/OCR/models/pixel_anchor/model.ckpt-173646'

dataset = sys.argv[1]
try:
    cfg_ckpt_path = sys.argv[3]
except:
    cfg_ckpt_path = '/workspace/OCR/models/s3d/model.ckpt-223826'
test_image_dir = '/workspace/OCR/datasets/TAL_OCR/190326_4122/%s/'%(dataset)
test_jsons = '/workspace/OCR/datasets/TAL_OCR/190326_4122/jsons/'
# test_image_dir = '/workspace/OCR/datasets/TAL_OCR/190505_2490/%s/'%(dataset)
# test_jsons = '/workspace/OCR/datasets/TAL_OCR/190505_2490/jsons/'
DATA_FORMAT = 'NHWC'
LABEL_TEXT = {
    'Background' : 0,
    '横式' : 1,
    '竖式' : 2,
    '脱式' : 3,
    '解方程' : 3
}
cfg_class_num = cfg.CLASS_NUM
select_threshold = 0.5
cfg_net_shape = (512, 512)
json_names = os.listdir(test_jsons)
image_datas = []
SHAPE = {}
for json_name in json_names:
    json_c_name = os.path.join(test_jsons, json_name)
    json_f = json.loads(open(json_c_name, encoding='utf-8').read(), encoding='bytes')
    image_datas_temp = json_f
    image_datas = image_datas + image_datas_temp



def get_rect(bbox):
    x1 = float(bbox[0][1:])
    y1 = float(bbox[1])
    x2 = float(bbox[2][1:])
    y2 = float(bbox[3])
    x3 = float(bbox[4][1:])
    y3 = float(bbox[5])
    x4 = float(bbox[6][1:])
    y4 = float(bbox[7])
    xmin = min([x1, x2, x3, x4])
    xmax = max([x1, x2, x3, x4])
    ymin = min([y1, y2, y3, y4])
    ymax = max([y1, y2, y3, y4])
    xmin = max([xmin, 0])
    ymin = max([ymin, 0])
    xmax = min([xmax, 1])
    ymax = min([ymax, 1])
    return ymin, xmin, ymax, xmax


def convert_json_to_dict(image_list):
    ground_truth_number = [0, 0, 0, 0] 
    for n in range(1, cfg.CLASS_NUM):
        ground_truth_dict[n] = {}
    for image_data in image_datas:
        image_name = image_data['pic_name'].split('/')[4]
        obj_l = {}
        for n in range(1, cfg.CLASS_NUM):
            obj_l[n] = []
        if image_name in image_list:
            final_boxes = []
            mark_datas = image_data['mark_datas']
            for mark_data in mark_datas:
                label = mark_data['label']
                children = label[0]["children"]
                if '文本' in children or '横式' in children:
                    c = '横式'
                else:
                    c = children[0]
                if c not in LABEL_TEXT:
                    continue
                bbox = mark_data['marked_path']
                if bbox == None:
                    continue
                bbox = bbox.split()
                if len(bbox) != 9:
                    continue
                # flag = True
                # break
                box = get_rect(bbox)
                final_boxes.append([c, box[0], box[1], box[2], box[3]])
            img = cv2.imread(test_image_dir + image_name)
            shape = img.shape
            for box in final_boxes:
                temp_label = int(LABEL_TEXT[box[0]])
                obj_l[temp_label].append([box[2] * shape[1], box[1] * shape[0], box[4] * shape[1], box[3] * shape[0], False])
                ground_truth_number[temp_label] += 1
                    
            for n in range(1, cfg.CLASS_NUM):
                ground_truth_dict[n][image_name] = obj_l[n]
        
    return ground_truth_number

def IOU_calculation(predict_box, ground_box):
    ground_box = [float(x) for x in ground_box]
    box_interaction = [max(ground_box[0], predict_box[0]), max(ground_box[1], predict_box[1]),
                       min(ground_box[2], predict_box[2]), min(ground_box[3], predict_box[3])]
    interaction_weight = box_interaction[2] - box_interaction[0]+1
    interaction_height = box_interaction[3] - box_interaction[1]+1
    if interaction_weight > 0 and interaction_height > 0:
        interaction_area = interaction_height * interaction_weight
        union_area = (ground_box[2] - ground_box[0]) * (ground_box[3] - ground_box[1]) + \
                    (predict_box[2] - predict_box[0]) * (predict_box[3] - predict_box[1]) - interaction_area
        return interaction_area / union_area
    else:
        return None
    
def calculate_precision_and_recall(groudth_dict, predict_result, the_whole_boxes, IOU_THRESHOLD):
    TP=[0]*len(predict_result)
    FP=[0]*len(predict_result)
    for index, predict_dict in enumerate(predict_result):
        predict_box = predict_dict["box"]
        name = predict_dict["image_name"]
        tmp_iou = -float('inf')
        tmpground_box =  []
        for ground_truth_box in groudth_dict[name]:
            middle_iou = IOU_calculation(predict_box[0:4], ground_truth_box[0:4])
            
            if middle_iou is not None:
                if middle_iou > tmp_iou:
                    tmp_iou = middle_iou
                    tmp_ground_box = ground_truth_box

        if tmp_iou >= IOU_THRESHOLD:
            if tmp_ground_box[-1] == False:
                TP[index] = 1
                tmp_ground_box[-1] = True
            else:
                FP[index]=1
        else:
            FP[index]=1

                
    recall = (sum(TP)+0.0) / (the_whole_boxes + 1e-10)
    precision = ((sum(TP)+0.0) / (sum(TP)+sum(FP) + 1e-10))
    return precision, recall


def produce_result(obj, image_name_list, p, r, l, select_thr = 0.5, nms_thr = 0.4):
    predict_result={}
    for n in range(1, cfg.CLASS_NUM):
        predict_result[n] = []
        
    for image_data in image_datas:
        image_name = image_data['pic_name'].split('/')[4]
        obj_l = {}
        if image_name in image_name_list:
            image_path = os.path.join(test_image_dir, image_name)
            if os.path.exists(image_path) is False:
                continue
            
            classes, scores, bboxes = obj.postprocess(p[image_name], r[image_name], l[image_name], select_threshold = select_thr, nms_threshold = nms_thr)
            
            shape = SHAPE[image_name]
            ssd_rs = get_det_result(classes, scores, bboxes)
            if cfg.MULTIPROCESS:
                try:
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    print("Image does not exist or is malformed. Please check the input image.")
                ssd_rs = postprocess_via_ratio(obj, img, ssd_rs, shape, select_threshold = select_thr, nms_threshold = nms_thr)
            result = {}
            for n in range(1, cfg.CLASS_NUM):
                result[n] = []               
            for bboxes in ssd_rs:
                if bboxes[0] in range(1, cfg.CLASS_NUM):
                    n+=1
                    ymin = int(bboxes[2] * shape[0])
                    xmin = int(bboxes[3] * shape[1])
                    ymax = int(bboxes[4] * shape[0])
                    xmax = int(bboxes[5] * shape[1])
                    result[bboxes[0]].append({"box": (xmin,ymin,xmax,ymax), "confidence": bboxes[1], "image_name": image_name}) 
            for n in range(1, cfg.CLASS_NUM):
                predict_result[n].extend(result[n])
                
    for n in range(1, cfg.CLASS_NUM):
        predict_result[n] = sorted(predict_result[n], key=lambda x: x["confidence"], reverse=True)
    return predict_result


if __name__ == "__main__":
    gpu_id = '0'
    obj = det_Layout(gpu_id, cfg_ckpt_path)
    image_name_list = os.listdir(test_image_dir)
    
    the_whole_boxes=convert_json_to_dict(image_name_list)

    iou_threshold = 0.5
    select_thr_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    nms_thr_list = [0.3, 0.4, 0.45, 0.5, 0.6, 0.7]
    map_dict={}

    print("**********************************")
    print('the whole number is :',np.array(the_whole_boxes)[range(1, cfg.CLASS_NUM)])
    print(cfg_ckpt_path)
    print("**********************************")
    n1 = the_whole_boxes[1]
    n2 = the_whole_boxes[2]
    n3 = the_whole_boxes[3]
    
    pr = {}
    l = {}
    b = {}
    
    for image_data in image_datas:
        image_name = image_data['pic_name'].split('/')[4]
        if image_name in image_name_list:
            image_path = os.path.join(test_image_dir, image_name)
            if os.path.exists(image_path) is False:
                continue
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            SHAPE[image_name] = img.shape
            p_t, l_t, b_t = obj.process_image(img)
            pr[image_name] = p_t
            l[image_name] = l_t
            b[image_name] = b_t
        
    
    if sys.argv[2] == 'select':
        nms_thr = 0.4
        for select_thr in select_thr_list:
    #     for nms_thr in nms_thr_list:
            tmp_predict = produce_result(obj, image_name_list, pr, l, b, select_thr, nms_thr)   
            tmp_ground_dict=copy.deepcopy(ground_truth_dict)
            p = []
            r = []
            f = []
            print("")
            for n in range(1, cfg.CLASS_NUM):
                precision, recall = calculate_precision_and_recall(tmp_ground_dict[n], tmp_predict[n], the_whole_boxes[n], iou_threshold)
                p.append(precision)
                r.append(recall)
            print('%.2f %.2f '%(select_thr, nms_thr), end = '')
            for P,R in zip(p,r):
                print('%.3f/%.3f/%.3f '%(P,R,2 * P* R / (P + R + 1e-10)), end = '')
        print("")
    else:
        select_thr = 0.7
        for nms_thr in nms_thr_list:
            tmp_predict = produce_result(obj, image_name_list, pr, l, b, select_thr, nms_thr)   
            tmp_ground_dict=copy.deepcopy(ground_truth_dict)
            p = []
            r = []
            print("")
            for n in range(1, cfg.CLASS_NUM):
                precision, recall = calculate_precision_and_recall(tmp_ground_dict[n], tmp_predict[n], the_whole_boxes[n], iou_threshold)
                p.append(precision)
                r.append(recall)
            print('%.2f %.2f '%(select_thr, nms_thr), end = '')
            for P,R in zip(p,r):
                print('%.3f/%.3f/%.3f '%(P,R,2 * P* R / (P + R + 1e-10)), end = '')
        print("")

