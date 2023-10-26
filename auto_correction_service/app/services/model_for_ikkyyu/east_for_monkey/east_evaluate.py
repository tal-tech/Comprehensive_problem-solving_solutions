# coding=utf-8
"""
EAST模型验证：
针对验证数据进行算法模型验证
1、对验证数据进行算法模型测试输出
2、对比模型输出和验证数据进行验证

"""

import os
import cv2
import shapely
import tqdm
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
from shapely.geometry import Polygon, MultiPoint  # 多边形

from east_inference import East

def read_label_file(file_path):
    """
    读取label文件，打印的bbox信息
    :param file_path:
    :param is_gt:
    :return:
    """
    with open(file_path, 'r') as f:
        # 读入所有文本行
        lines = f.readlines()
        # 容器每一行代表一个框(8个点)
        boxes_info_list = []

        # 遍历每行,进行切分
        for line in lines:
            info = line.split(",")
            bbox_info = []
            bbox_info.append(int(info[0]))
            bbox_info.append(int(info[1]))
            bbox_info.append(int(info[2]))
            bbox_info.append(int(info[3]))
            bbox_info.append(int(info[4]))
            bbox_info.append(int(info[5]))
            bbox_info.append(int(info[6]))
            bbox_info.append(int(info[7]))
            boxes_info_list.append(bbox_info)

    return boxes_info_list


def quad_iou(_gt_bbox, _pre_bbox):
    # 四边形四个点坐标的一维数组表示，[x,y,x,y....]

    gt_poly = Polygon(_gt_bbox).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    # print(Polygon(_gt_bbox).convex_hull)  # 可以打印看看是不是这样子

    pre_poly = Polygon(_pre_bbox).convex_hull
    # print(Polygon(_pre_bbox).convex_hull)

    union_poly = np.concatenate((_gt_bbox, _pre_bbox))  # 合并两个box坐标，变为8*2
    # print(MultiPoint(union_poly).convex_hull)  # 包含两四边形最小的多边形点

    if not gt_poly.intersects(pre_poly):  # 如果两四边形不相交
        iou = 0
        return iou
    else:
        try:
            inter_area = gt_poly.intersection(pre_poly).area  # 相交面积
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area) / (union_area-inter_area)  #错了
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
            return iou
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
            return iou


def evaluate(gt_boxes_list, pred_boxes_list, overlap=0.5):
    """
    评价网络模型:param gt_boxes_list: [[x1,y1,x2,y2,x3,y3,x4,y4],...]
    :param pred_boxes_list: [[x1,y1,x2,y2,x3,y3,x4,y4],...]
    :param overlap:
    :param r_overlap:
    :return:
    """
    TP = 0
    pred_flag = [0] * len(pred_boxes_list)
    gt_flag = [0] * len(gt_boxes_list)
    bad_boxes_info = []

    for i, pred_box_info in enumerate(pred_boxes_list):
        pre_bbox = np.array(pred_box_info).reshape(4, 2)

        max_iou = 0
        pair_gt_box_id = -1
        pair_gt_box = None

        for j, gt_box_info in enumerate(gt_boxes_list):

            if gt_flag[j] == 1:
                continue

            gt_bbox = np.array(gt_box_info).reshape(4, 2)  # 四边形二维坐标表示

            # 传入两个Bbox,返回iou
            iou = quad_iou(pre_bbox, gt_bbox)

            if iou > max_iou:
                max_iou = iou
                pair_gt_box_id = j
                pair_gt_box = gt_bbox

        if max_iou > overlap:
            TP += 1
            pred_flag[i] = 1
            if pair_gt_box_id != -1:
                gt_flag[pair_gt_box_id] = 1
        else:
            pred_bad_box = {}
            pred_bad_box['id'] = i
            pred_bad_box['bbox_pts'] = pre_bbox
            pred_bad_box['max_iou'] = max_iou

            if pair_gt_box_id != -1:
                pred_bad_box['pair_box_pts'] = pair_gt_box
            else:
                pred_bad_box['pair_box_pts'] = None
            bad_boxes_info.append(pred_bad_box)

    precision = TP / (float(len(pred_boxes_list)) + 1e-5)
    recall = TP / (float(len(gt_boxes_list)) + 1e-5)
    F1_score = 2 * (precision * recall) / (precision + recall + 1e-5)
    pred_boxes = float(len(pred_boxes_list))
    gt_boxes = float(len(gt_boxes_list))

    return TP, precision, recall, F1_score, pred_boxes, gt_boxes, bad_boxes_info

def evaluate_all(gt_txt_path, pre_txt_path, img_path, badcase_path):
    """
    测试指标
    :param gt_txt_path:
    :param pre_txt_path:
    :return:
    """
    global thr_min, thr_max, thr_interval

    # 读取gt下所有TXT文本
    gt_file_list = os.listdir(gt_txt_path)
    # 读取预测图片下TXT文本
    pred_file_list = os.listdir(pre_txt_path)
    print(len(gt_file_list), len(pred_file_list))
    assert len(pred_file_list) == len(gt_file_list), '{}和{}中的文件数目不一致'.format(gt_txt_path, pre_txt_path)

    all_TP = 0.0
    all_pred_num = 0.0
    all_gt_num = 0.0
    img_files_name = os.listdir(img_path)

    # 遍历所有预测文本,即对一个文本(一张图片)进行处理
    for pred_file in tqdm.tqdm(pred_file_list):

        # 若预测文本在gt文本中不存在
        if pred_file not in gt_file_list:
            assert 0, '{}预测文件没有在{}找到应gt文件'.format(pred_file, pre_txt_path)

        gt_bboxes_info_list = read_label_file(os.path.join(gt_txt_path, pred_file))
        pred_bboxes_info_list = read_label_file(os.path.join(pre_txt_path, pred_file))

        TP, precision, recall, F1_score, pred_boxes, gt_boxes, bad_boxes_info = evaluate(gt_bboxes_info_list,
                                                                                         pred_bboxes_info_list,
                                                                                         overlap=0.5)

        all_TP += TP
        all_gt_num += len(gt_bboxes_info_list)
        all_pred_num += len(pred_bboxes_info_list)

        if bad_boxes_info :
            basename = pred_file.split('.')[0]
            for img_name in img_files_name:
                if basename in img_name:
                    img = cv2.imread(os.path.join(img_path, img_name))
                    for bad_info in bad_boxes_info:
                        cv2.polylines(img, [bad_info['bbox_pts'].reshape((-1, 1, 2))], True, (0, 0, 255))
                        if bad_info['pair_box_pts'] is not None:
                            cv2.polylines(img, [bad_info['pair_box_pts'].reshape((-1, 1, 2))], True, (0, 255, 0))

                    cv2.imwrite(os.path.join(badcase_path, img_name), img)
                    break

    precision = all_TP / float(all_pred_num)
    recall = all_TP / float(all_gt_num)
    F1_score = 2 * (precision * recall) / (precision + recall)

    print("TP num:" + str(all_TP))
    print("all_gt_num num:" + str(all_gt_num))
    print("all_pred_num num:" + str(all_pred_num))

    print("precision:" + str(precision))
    print("recall:" + str(recall))
    print("F1_score:" + str(F1_score))

    return precision, recall, F1_score, [all_TP, all_gt_num, all_pred_num]


# 判断文件是否为有效（完整）的图片
# 输入参数为文件路径
# 会出现漏检的情况1
def is_valid_image(pathfile):
    bValid = True
    try:
        Image.open(pathfile).verify()
    except:
        bValid = False
    return bValid


if __name__ == "__main__":

    # checkpoint 文件路径
    checkpoint_path = ''

    # 测试图片文件夹路径
    val_img_dir = ''
    val_label_dir = ''

    # 算法测试结果和验证结果存储路径
    res_dir = ''

    # 读取文件夹下路径
    img_list = os.listdir(val_img_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print(ckpt)
    print('test models number:', len(ckpt.all_model_checkpoint_paths))
    print('start to evaluate east model...')

    models_test_res_list = []
    models_badcase_dir_list = []
    for model_path in ckpt.all_model_checkpoint_paths:
        model_name = model_path.split('/')[-1]

        model_res_dir = os.path.join(res_dir, model_name)

        # 创建当前模型的结果文件
        if os.path.exists(model_res_dir):
            shutil.rmtree(model_res_dir)
        os.makedirs(model_res_dir)

        # 结果图文件夹
        res_img_dir = os.path.join(model_res_dir, 'result_img')
        if os.path.exists(res_img_dir):
            shutil.rmtree(res_img_dir)
        os.makedirs(res_img_dir)

        # 结果txt文件
        res_txt_dir = os.path.join(model_res_dir, 'result_txt')
        if os.path.exists(res_txt_dir):
            shutil.rmtree(res_txt_dir)
        os.makedirs(res_txt_dir)
        models_test_res_list.append(res_txt_dir)

        # badcase路径
        badcase_dir = os.path.join(model_res_dir, 'badcase')
        if os.path.exists(badcase_dir):
            shutil.rmtree(badcase_dir)
        os.makedirs(badcase_dir)
        models_badcase_dir_list.append(badcase_dir)

        east = East(model_path)
        print(model_name, ' start to test....')
        for img_name in tqdm.tqdm(img_list):
            east.test(os.path.join(val_img_dir, img_name), res_txt_dir, res_img_dir, False)
        print('done')

    print('start to evaluate models....')
    for i, model_txt_dir in enumerate(models_test_res_list):
        print('eval ', model_txt_dir)
        precision, recall, F1_score, _ = evaluate_all(val_label_dir, model_txt_dir, val_img_dir, models_badcase_dir_list[i])
        with open(os.path.join(models_badcase_dir_list[i], 'result.txt'), 'w') as f:
            f.writelines('precision'+str(precision))
            f.writelines('recall'+str(recall))
            f.writelines('F1_score'+str(F1_score))
    print('done')


