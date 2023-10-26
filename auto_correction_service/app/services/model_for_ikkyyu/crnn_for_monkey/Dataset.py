from recognition_config import Config as config
from glob import glob
import os
import numpy as np
import cv2
from recognition_utils import label_replace_for_error, label_replace_for_train
import matplotlib.pyplot as plot
import skimage
from PIL import Image, ImageDraw, ImageFont
import subprocess
import re
import random
import math
from tqdm import tqdm


class DataSet(object):
    def __init__(self, train_data_list=config.TRAIN_DATA_LIST, val_data_list=config.VAL_DATA_LIST, model='crnn',
                 img_config=config.CRNN_IMG_CONFIG):
        self.train_data_list = [glob(os.path.join(train_data, '*')) for train_data in train_data_list]
        self.val_data_list = [glob(os.path.join(val_data, '*')) for val_data in val_data_list]
        self.model = model

    @staticmethod
    def list_to_sparse(label_list):
        '''
        讲list转换为tensorflow需要的稀疏矩阵
        :param label_list:
        :return:
        '''
        cfg = config()
        ont_hot = cfg.ONE_HOT
        index = []
        value = []
        max_length = 0
        batch_size = len(label_list)

        for x, labels in enumerate(label_list):
            if len(labels) > max_length:
                max_length = len(labels)
            for y, char in enumerate(labels):
                index.append([x, y])
                if ont_hot.get(char) == None:
                    print(char)
                value.append(ont_hot.get(char))

        shape = np.array([batch_size, max_length], dtype=np.int32)
        index = np.array(index, dtype=np.int32)

        value = np.array(value, dtype=np.int32)

        return [index, value, shape]

    def image_normal(self, image):
        img_config = config.CRNN_IMG_CONFIG
        if img_config['w_stable']:
            image = cv2.resize(image, (img_config['w_stable'], img_config['height']))
        else:
            h = img_config['height']
            w_max = img_config['w_max']
            w_min = img_config['w_min']

            if image.shape[0] != h:
                image = cv2.resize(image, (int(max(image.shape[1] / image.shape[0] * h, 1)), h))
            if image.shape[1] < w_min:
                image = cv2.resize(image, (w_min, h))
            if image.shape[1] > 250:
                image = cv2.resize(image, (w_max, h))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if img_config['normalization']:
            image = image / 255 * 2 - 1

        return image

    @staticmethod
    def get_labels(images_path):
        '''
        从label的文件夹中获得文件的标注，存入到list
        :param images_path:
        :return:
        '''
        label_list = []
        label_len = []

        for path in images_path:
            label = path.split('_')[-1].replace('.jpg', '')
            label = label.replace('.png', '')
            label = label.replace('.JPG', '')
            label = label.replace('.jpeg', '')
            label = label_replace_for_error(label)
            label = label_replace_for_train(label)
            label_list.append(label)
            label_len.append(len(label))

        return label_list, label_len


    def get_crnn_label(self, images_path):
        '''
        讲label转换为crnn需要的类型
        :param images_path:
        :return:
        '''

        label_list, label_len = self.get_labels(images_path)

        labels = self.list_to_sparse(label_list.copy())
        label_len = np.array(label_len, dtype=np.int32)

        return labels, label_len, label_list

    def data_enhance(self, img):
        img_list = []
        img1 = skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True) * 255  # 高斯噪声
        img1 = np.asarray(img1, np.uint8)
        img1 = self.image_normal(img1)
        img2 = skimage.util.random_noise(img, mode='salt', seed=None, clip=True) * 255  # 椒盐噪声
        img2 = np.asarray(img2, np.uint8)
        img2 = self.image_normal(img2)
        img_list.append(img1)
        img_list.append(img2)
        return img_list

    def get_imges(self, images_path):
        image_list = []
        max_wide = 0
        images_wide = []

        for path in images_path:
            image = cv2.imread(path)
            image_enhance = image.copy()

            image = self.image_normal(image)

            images_wide.append(image.shape[1])
            image_list.append(image)

            if config.DATA_ENHANCE:
                img_list = self.data_enhance(image_enhance)
                img_list.extend(img_list)
            if image.shape[1] > max_wide:
                max_wide = image.shape[1]

        images = np.zeros([len(image_list), config.IMAGE_HEIGHT, max_wide])

        for i, image in enumerate(image_list):
            images[i, :, 0:image.shape[1]] = image
        images = images[..., np.newaxis]

        wides = np.array(images_wide, dtype=np.int32)

        return images, wides

    def get_all_inputs(self, images_path):
        images, wides = self.get_imges(images_path)
        labels, length, real_labels = self.get_crnn_label(images_path)

        return images, labels, wides, length, real_labels

    def trian_generator(self):

        all_data_list = self.train_data_list
        data_batch = config.TRAIN_DATA_BATCH

        step = [0] * len(all_data_list)
        epoch = [0] * len(all_data_list)
        while True:
            images_path = []
            for i, all_data in enumerate(all_data_list):

                batch_size = data_batch[i]
                if (step[i] + 1) * batch_size > len(all_data):
                    random.shuffle(all_data)
                    step[i] = 0
                    epoch[i] = epoch[i] + 1

                images_path.extend(all_data[step[i] * batch_size:(step[i] + 1) * batch_size])
                step[i] = step[i] + 1

            yield self.get_all_inputs(images_path) + (epoch,)

    def display_generator(self):
        '''
        按类别返回验证集的数据
        :return:
        '''
        train_data_list = self.train_data_list
        val_data_list = self.val_data_list
        batch_size = config.BATCH_SIZE
        while True:
            train_inputs = []
            val_inputs = []

            # 每种数据合并在一起的数据
            train_merge_path = []
            val_merge_path = []
            for i, train_data in enumerate(train_data_list):
                train_images_path = random.sample(train_data, batch_size)
                train_merge_path.extend(random.sample(train_data, config.TRAIN_DATA_BATCH[i]))
                train_inputs.append(self.get_all_inputs(train_images_path))

            train_inputs.append(self.get_all_inputs(train_merge_path))

            for i, val_data in enumerate(val_data_list):
                val_images_path = random.sample(val_data, batch_size)
                val_merge_path.extend(random.sample(train_data, config.VAL_DATA_BATCH[i]))
                val_inputs.append(self.get_all_inputs(val_images_path))

            val_inputs.append(self.get_all_inputs(val_merge_path))

            yield train_inputs, val_inputs

    def create_val_data(self):
        val_data_list = self.val_data_list

        all_val_data = []

        print('process validation data：')
        for num, val_data in enumerate(val_data_list):
            batch = config.VAL_DATA_BATCH[num]
            val_data_list = []
            print('process {} batch：'.format(num + 1))
            # while i*batch<len(val_data):
            for i in tqdm(range(math.ceil(len(val_data) / batch))):
                if (i + 1) * batch > len(val_data):
                    end = len(val_data)
                else:
                    end = (i + 1) * batch

                val_data_list.append(self.get_all_inputs(val_data[i * batch:end]) + (val_data[i * batch:end],))
            all_val_data.append(val_data_list)

        return all_val_data


def label_clean():
    '''

    :return:
    '''

    train_data_list = [glob(os.path.join(train_data, '*')) for train_data in config.TRAIN_DATA_LIST]
    val_data_list = [glob(os.path.join(val_data, '*')) for val_data in config.VAL_DATA_LIST]
    all_path = []
    for path in train_data_list + val_data_list:
        all_path.extend(path)

    for path in tqdm(all_path):
        label = path.split('/')[-1].split('_')[-1].split('.')[0]
        label = label_replace_for_train(label)
        label = label_replace_for_error(label)

        if not label:
            print(path)
            subprocess.call(['rm', path])
            continue

        img = cv2.imread(path)
        if not img.any():
            print(path)
            subprocess.call(['rm', path])
            continue

        length = max(int(img.shape[1] / img.shape[0] * 32), 10)
        length = ((length - 2) / 2 - 2) / len(label)
        if length < 1:
            print(path)
            subprocess.call(['rm', path])
            continue