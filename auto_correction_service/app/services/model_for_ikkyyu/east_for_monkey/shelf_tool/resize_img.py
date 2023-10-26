# coding:utf-8
import os
import cv2
from tqdm import tqdm


def resize_img(file_path, tar_path, scale_size):
    if os.path.exists(file_path):
        try:
            x = cv2.imread(file_path)
            shape = x.shape
        except:
            print(file_path)

        if scale_size != 0:  # 如果进行缩放，以短边为基准等比例缩放，如果要以长边为基准，将< 变成 >
            if shape[0] < shape[1]:
                x = cv2.resize(x, (int(shape[1] / shape[0] * scale_size), scale_size))
            else:
                x = cv2.resize(x, (scale_size, int(shape[0] / shape[1] * scale_size)))
        cv2.imwrite(tar_path, x)


def resize_all_img(file_dir, tar_dir, scale_size):
    file_list = os.listdir(file_dir)  # 该文件夹下所有的文件（包括文件夹）
    for file in tqdm(file_list):  # 遍历所有文件
        file_path = os.path.join(file_dir, file)
        file_tar_path = os.path.join(tar_dir, file)
        resize_img(file_path, file_tar_path, scale_size)


# 主函数
if __name__ == '__main__':
    file_dir = "/workspace/boby/project_git/pipeline_for_ikkyyu/20190531_81_test"
    tar_dir = '/workspace/boby/data/ocr_test_data/20190531_81_sinianji_test/data'
    scale_size = 720
    resize_all_img(file_dir, tar_dir, scale_size)
