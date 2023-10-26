# coding=utf-8

import os
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形
import tqdm
from PIL import Image
from PIL import ImageDraw
import shutil



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

        # pre_bbox = [[pred_box_info[0], pred_box_info[1]],
        #             [pred_box_info[2], pred_box_info[3]],
        #             [pred_box_info[4], pred_box_info[5]],
        #             [pred_box_info[6], pred_box_info[7]]]
        pre_bbox = np.array(pred_box_info).reshape(4, 2)

        max_iou = 0
        pair_gt_box_id = -1
        pair_gt_box = []

        for j, gt_box_info in enumerate(gt_boxes_list):

            if gt_flag[j] == 1:
                continue

            # gt_bbox = [[gt_box_info[0], gt_box_info[1]],
            #           [gt_box_info[2], gt_box_info[3]],
            #           [gt_box_info[4], gt_box_info[5]],
            #           [gt_box_info[6], gt_box_info[7]]]
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
                pred_bad_box['pair_box_pts'] = []
            bad_boxes_info.append(pred_bad_box)

    precision = TP / (float(len(pred_boxes_list)) + 1)
    recall = TP / (float(len(gt_boxes_list)) + 1)
    F1_score = 2 * (precision * recall) / (precision + recall + 1)
    pred_boxes = float(len(pred_boxes_list)) + 1
    gt_boxes = float(len(gt_boxes_list)) + 1

    return TP, precision, recall, F1_score, pred_boxes, gt_boxes, bad_boxes_info


def evaluate_all(gt_txt_path, pre_txt_path):
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

    assert len(pred_file_list) == len(gt_file_list), '{}和{}中的文件数目不一致'.format(gt_txt_path, pre_txt_path)

    all_TP = 0.0
    all_pred_num = 0.0
    all_gt_num = 0.0
    per_file_res = []

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
        res = {"bad_boxes_info": bad_boxes_info}

    precision = all_TP / float(all_pred_num)
    recall = all_TP / float(all_gt_num)
    F1_score = 2 * (precision * recall) / (precision + recall)

    print("TP num:" + str(all_TP))
    print("all_gt_num num:" + str(all_gt_num))
    print("all_pred_num num:" + str(all_pred_num))

    print("precision:" + str(precision))
    print("recall:" + str(recall))
    print("F1_score:" + str(F1_score))


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


def draw_pre_result_to_gt_img(pre_txt_dir, test_gt_img_dir, save_pre_result_gt_img_dir):

    if os.path.exists(save_pre_result_gt_img_dir):
        shutil.rmtree(save_pre_result_gt_img_dir)
    os.makedirs(save_pre_result_gt_img_dir)

    # 读取预测图片下TXT文本
    pred_file_list = os.listdir(pre_txt_dir)
    test_gt_img_list = os.listdir(test_gt_img_dir)

    # 遍历所有预测文本,即对一个文本(一张图片)进行处理
    for pred_file in tqdm.tqdm(pred_file_list):

        for test_gt in test_gt_img_list:

            if pred_file.split('.')[0] == test_gt.split('.')[0]:

                # 检测图片在本地是否存在
                img_path = os.path.join(test_gt_img_dir, test_gt)
                if not os.path.exists(img_path):
                    print("\n")
                    print("Bobby: not exist image is " + img_path)
                    continue

                # 检测文件是否能正常打开
                if is_valid_image(img_path) is False:
                    print("\n")
                    print("Bobby: error image is " + img_path)
                    continue

                # 打开图片,并对图片进行处理
                with Image.open(img_path) as im:
                    show_gt_im = im.copy()
                    # 画图
                    draw = ImageDraw.Draw(show_gt_im)
                    # 创建文件生成图片对应的TXT,并保存边框数据
                    bbox_list = read_label_file(os.path.join(pre_txt_dir, pred_file))
                    for bb in bbox_list:
                        xy_list = np.array(bb).reshape(4, 2)  # 四边形二维坐标表示
                        draw.line([tuple(xy_list[0]),
                                   tuple(xy_list[1]),
                                   tuple(xy_list[2]),
                                   tuple(xy_list[3]),
                                   tuple(xy_list[0])],
                                  width=3, fill='red')
                    show_gt_im.save(os.path.join(save_pre_result_gt_img_dir, test_gt))


if __name__ == "__main__":
    gt_txt_path = '/workspace/boby/data/ocr_test_data/xcs_190505_100/test_label/'
    # gt_txt_path = '/workspace/boby/data/ocr_test_data/toc_190408/test_labels_720p/'
    # gt_txt_path = '/workspace/boby/data/ocr_test_data/xcs_190505_100/test_label/'
    pre_txt_path = '/workspace/boby/data/ocr_test_data/xcs_190505_100/test_result_EAST_tf_190513_1/result_txt/'
    # test_gt_img_dir='/workspace/boby/projects/EAST_tf/data/20190326_xcs_mark_3000/test_gt'
    # save_pre_result_gt_img_dir='/workspace/boby/projects/EAST_tf/data/20190326_xcs_mark_3000/pre_result_gt_img'
    evaluate_all(gt_txt_path, pre_txt_path)
    # draw_pre_result_to_gt_img(pre_txt_path,test_gt_img_dir,save_pre_result_gt_img_dir)