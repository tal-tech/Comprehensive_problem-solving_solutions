# coding=utf-8
import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np


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


def get_rect(bbox):
    x1 = float(bbox[0][1:])
    y1 = float(bbox[1])
    x2 = float(bbox[2][1:])
    y2 = float(bbox[3])
    x3 = float(bbox[4][1:])
    y3 = float(bbox[5])
    x4 = float(bbox[6][1:])
    y4 = float(bbox[7])
    xy = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return xy


def preprocess(json_dir,img_dir,save_txt_dir,d_wight,d_height):
    # 源图片与对应json文件路径

    if not os.path.exists(save_txt_dir):
        os.mkdir(save_txt_dir)

    # 读取json所有文件
    json_names = os.listdir(json_dir)
    print("\n")
    print('Bobby:found %d origin jsons.' % len(json_names))

    # 遍历所有的json文件,将有用信息合并到images_datas
    images_datas = []
    for json_name in json_names:
        json_path = os.path.join(json_dir, json_name)
        json_file = json.loads(open(json_path, encoding='utf-8').read(), encoding='bytes')
        # 将json文件中 true_message 的 image_datas 字段读取出来
        images_datas_temp = json_file['true_message']['image_datas']
        images_datas = images_datas + images_datas_temp

    # 遍历jsonList进行数据预处理
    for image_datas in tqdm(images_datas):

        # 取出图片名字
        o_img_fname = image_datas['pic_name'].split('/')[-1]

        # 检测图片在本地是否存在
        if not os.path.exists(os.path.join(img_dir, o_img_fname)):
            print("\n")
            print("Bobby: not exist image is " + os.path.join(img_dir, o_img_fname))
            continue

        # 检测文件是否能正常打开
        if is_valid_image(os.path.join(img_dir, o_img_fname)) is False:
            print("\n")
            print("Bobby: error image is " + os.path.join(img_dir, o_img_fname))
            continue

        # 获取训练图片设定 宽 与 高
        anno_list = image_datas['mark_datas']

        # 创建文件生成图片对应的TXT,并保存边框数据
        txt_file = os.path.join(save_txt_dir, os.path.splitext(o_img_fname)[0] + '.txt')

        with open(txt_file, 'w') as f:
            for anno, i in zip(anno_list, range(len(anno_list))):

                if 'marked_path' not in anno:
                    continue
                bbox = anno['marked_path']
                bbox = bbox.split()
                xy_list = get_rect(bbox)

                xy_list[:, 0] = xy_list[:, 0] * d_wight
                xy_list[:, 1] = xy_list[:, 1] * d_height


                # 样例: 374,1,494,0,492,85,372,86,###
                line_label = "{},{},{},{},{},{},{},{},###".format(str(int(xy_list[0][0])), str(int(xy_list[0][1])),
                                                                  str(int(xy_list[1][0])), str(int(xy_list[1][1])),
                                                                  str(int(xy_list[2][0])), str(int(xy_list[2][1])),
                                                                  str(int(xy_list[3][0])), str(int(xy_list[3][1])))

                f.write(line_label + '\n')



if __name__ == '__main__':
    #json所在文件夹
    json_dir='/workspace/boby/projects/EAST_tf/data/20190326_xcs_mark_3000/json/'
    # 图片所在文件夹
    img_dir='/workspace/boby/projects/EAST_tf/data/20190326_xcs_mark_3000/orig_test_imgs/'
    # TXT保存文件夹
    save_txt_dir='/workspace/boby/projects/EAST_tf/data/20190326_xcs_mark_3000/test_txt/'
    # 缩放大小
    d_wight=1280
    d_height=1280
    preprocess(json_dir,img_dir,save_txt_dir,d_wight,d_height)
