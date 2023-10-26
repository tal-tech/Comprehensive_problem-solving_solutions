####################
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from preprocessing.ssd_vgg_preprocessing import preprocess_for_eval
from preprocessing import ssd_vgg_preprocessing
from nets import np_methods, re_shuffle_net
from notebooks import visualization
import time

##############################config info###############################
DATA_FORMAT = 'NHWC'
prediction_fun = tf.sigmoid
cfg_class_num = 21
select_threshold = 0.2
cfg_net_shape = (320, 320)
cfg_img_path  = 'E:\\ReSuffleNet_300\\demo\\111.jpg'
cfg_ckpt_path = 'E:\\ReSuffleNet_300\\checkpoints\\model.ckpt-333372'

cfg_cls_name_body = {'0': 'background', '1': 'person'}
cfg_cls_name_voc  = {'0': 'background', '1': 'aeroplane', '2': 'bicycle', '3': 'bird', '4': 'boat', '5': 'bottle',
                     '6': 'bus', '7': 'car', '8': 'cat', '9': 'chair', '10': 'cow', '11': 'diningtable', '12': 'dog',
                     '13': 'horse', '14': 'motorbike', '15': 'person', '16': 'pottedplant', '17': 'sheep', '18': 'sofa',
                     '19': 'train', '20': 'tvmonitor'}


cfg_cls_name_coco = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motocycle', '5': 'airplane',
                     '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant',
                     '12': 'wrong', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat',
                     '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear',
                     '24': 'zebra', '25': 'giraffe', '26': 'wrong', '27': 'backpack', '28': 'umbrella', '29': 'wrong',
                     '30': 'wrong', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis',
                     '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove',
                     '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '45': 'wrong',
                     '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl',
                     '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot',
                     '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch',
                     '64': 'pottedplant', '65': 'bed', '66': 'wrong', '67': 'diningtable', '68':'wrong', '69': 'wrong',
                     '70': 'toilet', '71': 'wrong', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote',
                     '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster',
                     '81': 'sink', '82': 'refrigerator', '83': 'wrong', '84': 'book', '85': 'clock', '86': 'vase',
                     '87': 'scissors', '88': 'teddy bear', '89': 'hair drier', '90': 'toothbrush', '91': 'wrong',
                     '92': 'wrong', '93': 'wrong', '94': 'wrong', '95': 'wrong'}
##############################config info###############################


def detection(img_path):
    image_ph  = tf.placeholder(tf.float32, shape = (None, None, 3))
    img_input, glabels, gbboxes, gbbox_img = \
        preprocess_for_eval(image_ph, None, None,
                            cfg_net_shape, DATA_FORMAT,
                            resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    img_input_ext = tf.expand_dims(img_input, 0)

    # net
    ssd_params = re_shuffle_net.SSDNet.default_params._replace(
        num_classes=cfg_class_num)
    ssd = re_shuffle_net.SSDNet(ssd_params)

    ssd_anchors = ssd.anchors(ssd_params.img_shape)
    with slim.arg_scope(ssd.arg_scope(data_format=DATA_FORMAT)):
         predictions, localisations, logits, end_points = ssd.net(img_input_ext,
                                                                  is_training=False,
                                                                  prediction_fn=prediction_fun)
    # detection
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, cfg_ckpt_path)

        bgr_im = cv2.imread(img_path)
        rgb_im = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2RGB)

        for ii in range(1):
            start = time.time()
            imgg, predictions, localisations, rbbox_img = sess.run([img_input, predictions, localisations, gbbox_img],
                                                                   feed_dict={image_ph: rgb_im})
            elapsed = time.time()
            elapsed = elapsed - start
            print('time %.5f' %elapsed)

        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(predictions, localisations,
                                                                  ssd_anchors, select_threshold=select_threshold,
                                                                  img_shape=cfg_net_shape, num_classes=cfg_class_num,
                                                                  decode=True)
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.5)

        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        visualization.plt_bboxes(rgb_im, rclasses, cfg_cls_name_voc, rscores, rbboxes)


if __name__ == '__main__':
    detection(cfg_img_path)

