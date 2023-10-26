import os
import shutil
import tqdm
import json
import cv2
# import pandas as pd
import numpy as np

from east_inference import East
from east_evaluate import evaluate, read_label_file

def make_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def read_json_file(json_path):
    boxes_dict = json.loads(open(json_path, encoding='utf-8').read(), encoding='bytes')
    boxes_list = []
    for box_dict in boxes_dict:
        boxes_list.append(box_dict['bbox'])

    return boxes_list

def clear_boxes(boxes_list):
    """
    去除boxes中的横线框
    :param boxes_list:
    :return:
    """
    np_boxes = np.array(boxes_list).reshape([-1, 4, 2])

    for i in range(np_boxes.shape[0]):
        np_boxes[i] = cv2.boxPoints(cv2.minAreaRect(np_boxes[i]))

    boxes_h = np.sqrt(np.square(np_boxes[:, 0, 1] - np_boxes[:, 3, 1]) + np.square(np_boxes[:, 0, 0] - np_boxes[:, 3, 0]))
    boxes_w = np.sqrt(np.square(np_boxes[:, 0, 1] - np_boxes[:, 1, 1]) + np.square(np_boxes[:, 0, 0] - np_boxes[:, 1, 0]))
    avg_h = np.sum(boxes_h) / len(boxes_h)
    boxes = np_boxes[np.where(np.logical_and(boxes_h > (avg_h/3), boxes_w < boxes_h*3))].reshape([-1, 8])

    return np.array(boxes, dtype=np.int).tolist()

def is_line_boxes(boxes, ratio=4, h_size=10):
    boxes = np.array(boxes).reshape([4, 2])
    np_boxes = cv2.boxPoints(cv2.minAreaRect(boxes))

    boxes_h = np.sqrt(
        np.square(np_boxes[0, 1] - np_boxes[3, 1]) + np.square(np_boxes[0, 0] - np_boxes[3, 0]))
    boxes_w = np.sqrt(
        np.square(np_boxes[0, 1] - np_boxes[1, 1]) + np.square(np_boxes[0, 0] - np_boxes[1, 0]))

    if boxes_w/boxes_h > 6:
        return True
    else:
        if (boxes_w/boxes_h > ratio) and (boxes_h<=h_size):
            return True
        else:
            return False

def test_train_data():
    test_dir = '/share/zzh/raw_data'
    dirs_list = os.listdir(test_dir)

    CKPT_PATH = '/share/zzh/east_data/ckpt/v0.7.11/east_model_V0.7.11'
    east = East(CKPT_PATH)
    east.write_img = True

    for target_dir in dirs_list:
        print('start to detection ', target_dir, ' dir')
        img_path = os.path.join(test_dir, target_dir, 'imgs')

        res_txt_path = os.path.join(test_dir, target_dir, 'res_txt')
        res_img_path = os.path.join(test_dir, target_dir, 'res_img')
        make_dir(res_txt_path)
        make_dir(res_img_path)

        img_list = os.listdir(img_path)
        for img_name in tqdm.tqdm(img_list):
            east.test(os.path.join(img_path, img_name), res_txt_path, res_img_path, dispaly_print=False)

def find_dirty_data():
    test_dir = '/share/zzh/raw_data'
    dirs_list = os.listdir(test_dir)

    f = open('/share/zzh/dirty_data.txt', 'w')

    for target_dir in dirs_list:
        print('start to eval ', target_dir, ' dir')
        label_path = os.path.join(test_dir, target_dir, 'jsons')
        res_txt_path = os.path.join(test_dir, target_dir, 'res_txt')
        img_path = os.path.join(test_dir, target_dir, 'imgs')
        img_name_list = os.listdir(img_path)

        bad_case_dir = os.path.join(test_dir, target_dir, 'badcase')
        make_dir(bad_case_dir)

        res_txt_list = os.listdir(res_txt_path)

        for txt_name in tqdm.tqdm(res_txt_list):
            json_name = txt_name.split('.')[0] + '.json'
            if not os.path.exists(os.path.join(label_path, json_name)):
                print(os.path.join(label_path, json_name), ' not exist!')
                continue

            gt_bboxes = read_label_file(os.path.join(res_txt_path, txt_name))
            pd_bboxes = read_json_file(os.path.join(label_path, json_name))

            # gt_bboxes = clear_boxes(gt_bboxes)
            # pd_bboxes = clear_boxes(pd_bboxes)

            res = evaluate(gt_bboxes, pd_bboxes, 0.3)
            bad_boxes_info = res[-1]
            not_pair_pd_box_num = 0
            if bad_boxes_info:
                basename = txt_name.split('.')[0]
                for img_name in img_name_list:
                    if basename in img_name:
                        img = cv2.imread(os.path.join(img_path, img_name))
                        for bad_info in bad_boxes_info:
                            if is_line_boxes(bad_info['bbox_pts']):
                                continue
                            cv2.polylines(img, [bad_info['bbox_pts'].reshape((-1, 1, 2))], True, (0, 0, 255))
                            if bad_info['pair_box_pts'] is not None:
                                cv2.polylines(img, [bad_info['pair_box_pts'].reshape((-1, 1, 2))], True, (0, 255, 0))
                            else:
                                not_pair_pd_box_num += 1
                        cv2.imwrite(os.path.join(bad_case_dir, img_name), img)

                        w_str = os.path.join(img_path, img_name) + ',' + str(not_pair_pd_box_num) + '\r\n'
                        f.writelines(w_str)
                        break
    f.close()




if __name__ == "__main__":
    # test_train_data()
    find_dirty_data()
