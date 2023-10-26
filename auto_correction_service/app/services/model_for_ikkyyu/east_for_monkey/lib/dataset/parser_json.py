
# coding:utf-8
import os
import cv2
import json
from urllib import request
import numpy as np
from PIL import Image
from tqdm import tqdm

RAW_JSONS_DIR = '/root/raw_json/v4'
IMG_DIR = '/share/zzh/raw_data/4/imgs'
JSON_DIR = '/share/zzh/raw_data/4/jsons'
SHOW_DIR = '/share/zzh/raw_data/4/show'


def need_img_name(txt_path):
    with open(txt_path, 'r') as f:
        img_names = [name for name in f.readlines()]
    return img_names

def get_rect(bbox):
    x1 = float(bbox[0][1:])
    y1 = float(bbox[1])
    x2 = float(bbox[2][1:])
    y2 = float(bbox[3])
    x3 = float(bbox[4][1:])
    y3 = float(bbox[5])
    x4 = float(bbox[6][1:])
    y4 = float(bbox[7])
    xy = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
    return xy

def parser_jsons():
    json_files_path = os.listdir(RAW_JSONS_DIR)

    images_data = []
    for json_file_name in json_files_path:
        if 'json' in json_file_name:
            json_path = os.path.join(RAW_JSONS_DIR, json_file_name)
            json_data = json.loads(open(json_path, encoding='utf-8').read(),
                                    encoding='bytes')
            if 'true_message' in json_data:
                images_data_temp = json_data['true_message']['image_datas']
            else:
                images_data_temp = json_data['image_datas']
            images_data += images_data_temp

    need_img_names = need_img_name('/root/raw_json/v4/imgs_names.txt')
    for img_data in tqdm(images_data):
        # print(img_data)
        # if not img_data:
        #     continue
        # if not img_data['mark_datas']:
        #     continue
        for txt_name in need_img_names:
            name = txt_name.replace('\n', '')
            if name in img_data['pic_name']:
                # print(name)
                need_img_names.remove(txt_name)

                mark_datas = img_data['mark_datas']
                url = img_data['pic_url']
                img_name = img_data['pic_name'].split('/')[3]
                img_save_path = os.path.join(IMG_DIR, img_name)
                request.urlretrieve(url, img_save_path)
                angle = int(img_data['rotate_angle'])
                img_data_dict_list = []

                if angle == 180:

                    origin = Image.open(img_save_path)
                    rotated = origin.rotate(180, expand=1)
                    rotated.save(os.path.join(img_save_path))
                    width, height = rotated.size
                    show_img = cv2.imread(img_save_path)
                    for iter, mark_data in enumerate(mark_datas):
                        if not('横式' in mark_data['label'][0]['children'] or '文本' in mark_data['label'][0]['children']):
                            continue
                        img_data_dict = {}
                        box = get_rect(mark_data['marked_path'].split())
                        true_box = np.array([1 - box[4], 1 - box[5], 1 - box[6], 1 - box[7],
                                    1 - box[0], 1 - box[1], 1 - box[2], 1 - box[3]]).reshape([4, 2])
                        true_box[:, 0] = true_box[:, 0] * width
                        true_box[:, 1] = true_box[:, 1] * height
                        true_box = true_box.astype(np.int32).reshape([8, ])
                        img_data_dict['bbox'] = true_box.tolist()
                        img_data_dict['label'] = mark_data['marked_text']
                        img_data_dict_list.append(img_data_dict)
                        cv2.polylines(show_img,
                                      [true_box.reshape((-1, 1, 2))],
                                      True,
                                      (0, 255, 0))
                    cv2.imwrite(os.path.join(SHOW_DIR, img_name), show_img)
                elif angle == 90:
                    origin = Image.open(img_save_path)
                    rotated = origin.rotate(270, expand=1)
                    rotated.save(os.path.join(img_save_path))
                    width, height = rotated.size
                    show_img = cv2.imread(img_save_path)

                    for iter, mark_data in enumerate(mark_datas):
                        img_data_dict = {}
                        box = get_rect(mark_data['marked_path'].split())
                        true_box = np.array([1 - box[7], box[6], 1 - box[1], box[0],
                                            1 - box[3], box[2], 1 - box[5], box[4]]).reshape([4, 2])
                        true_box[:, 0] = true_box[:, 0] * width
                        true_box[:, 1] = true_box[:, 1] * height
                        true_box = true_box.astype(np.int32).reshape([8, ])
                        img_data_dict['bbox'] = true_box.tolist()
                        img_data_dict['label'] = mark_data['marked_text']
                        img_data_dict_list.append(img_data_dict)
                        cv2.polylines(show_img,
                                      [true_box.reshape((-1, 1, 2))],
                                      True,
                                      (0, 255, 0))
                    cv2.imwrite(os.path.join(SHOW_DIR, img_name), show_img)
                elif angle == 270:
                    origin = Image.open(img_save_path)
                    rotated = origin.rotate(90, expand=1)
                    rotated.save(os.path.join(img_save_path))
                    width, height = rotated.size
                    show_img = cv2.imread(img_save_path)

                    for iter, mark_data in enumerate(mark_datas):
                        img_data_dict = {}
                        box = get_rect(mark_data['marked_path'].split())
                        true_box = np.array([box[3], 1 - box[2], box[5], 1 - box[4],
                                             box[7], 1 - box[6], box[1], 1 - box[0]]).reshape([4, 2])
                        true_box[:, 0] = true_box[:, 0] * width
                        true_box[:, 1] = true_box[:, 1] * height
                        true_box = true_box.astype(np.int32).reshape([8, ])
                        img_data_dict['bbox'] = true_box.tolist()
                        img_data_dict['label'] = mark_data['marked_text']
                        img_data_dict_list.append(img_data_dict)
                        cv2.polylines(show_img,
                                  [true_box.reshape((-1, 1, 2))],
                                  True,
                                  (0, 255, 0))
                    cv2.imwrite(os.path.join(SHOW_DIR, img_name), show_img)
                elif angle == 0:
                    img = cv2.imread(img_save_path)
                    height, width, _ = img.shape
                    show_img = cv2.imread(img_save_path)

                    for iter, mark_data in enumerate(mark_datas):
                        img_data_dict = {}
                        box = get_rect(mark_data['marked_path'].split())
                        true_box = np.array([box[0], box[1], box[2], box[3],
                                             box[4], box[5], box[6], box[7]]).reshape([4, 2])
                        true_box[:, 0] = true_box[:, 0] * width
                        true_box[:, 1] = true_box[:, 1] * height
                        true_box = true_box.astype(np.int32).reshape([8, ])
                        img_data_dict['bbox'] = true_box.tolist()
                        img_data_dict['label'] = mark_data['marked_text']
                        img_data_dict_list.append(img_data_dict)
                        cv2.polylines(show_img,
                                      [true_box.reshape((-1, 1, 2))],
                                      True,
                                      (0, 255, 0))
                    cv2.imwrite(os.path.join(SHOW_DIR, img_name), show_img)
                else:
                    print(img_name, 'has error angle')
                    continue

                base_name = img_name.split('.')[0]
                # print(img_data_dict_list)
                with open(os.path.join(JSON_DIR, base_name+'.json'), 'w') as f:
                    f.write(json.dumps(img_data_dict_list, indent=4, ensure_ascii=False))
                    f.close()



if __name__ == '__main__':

    parser_jsons()
    a = np.zeros([2,2]).tolist()
