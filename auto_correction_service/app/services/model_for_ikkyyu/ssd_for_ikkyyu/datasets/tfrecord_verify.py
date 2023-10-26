###########################################
####################li bing################
###########################################
import os,cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def verify(split_name):
    BASE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    #TFRECORD_PATH = 'E:\\train data\\VOC2012\\tfrecords'
    TFRECORD_PATH = 'E:\\body_dataset_for_detection_new\\tfrecords'
    FILE_PATTERN  = os.path.join(TFRECORD_PATH, 'voc_%s.record' % split_name)
    NUMBER_SAMPLE = 120000
    NUMBER_CLASSES= 80

    _keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        # Object boxes and classes.
        'image/object/bbox/xmin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(tf.int64),
        'image/object/class/text':
            tf.VarLenFeature(tf.string),
        'image/object/area':
            tf.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.VarLenFeature(tf.int64),
        'image/object/difficult':
            tf.VarLenFeature(tf.int64),
        'image/object/group_of':
            tf.VarLenFeature(tf.int64),
        'image/object/weight':
            tf.VarLenFeature(tf.float32),}

    _items_to_handlers = {
        'image':
            slim.tfexample_decoder.Image(
                image_key='image/encoded', format_key='image/format', channels=3),
        # fields.InputDataFields.source_id: (
        #     slim.tfexample_decoder.Tensor('image/source_id')),
        # fields.InputDataFields.key: (
        #     slim.tfexample_decoder.Tensor('image/key/sha256')),
        # fields.InputDataFields.filename: (
        #     slim.tfexample_decoder.Tensor('image/filename')),
        # Object boxes and classes.
        'groundtruth_boxes': (
            slim.tfexample_decoder.BoundingBox(['xmin', 'ymin', 'xmax', 'ymax'],
                                                'image/object/bbox/')),
        # fields.InputDataFields.groundtruth_area:
        #     slim.tfexample_decoder.Tensor('image/object/area'),
        # fields.InputDataFields.groundtruth_is_crowd: (
        #     slim.tfexample_decoder.Tensor('image/object/is_crowd')),
        # fields.InputDataFields.groundtruth_difficult: (
        #     slim.tfexample_decoder.Tensor('image/object/difficult')),
        'groundtruth_classes': (
            slim.tfexample_decoder.Tensor('image/object/class/label')),
        # fields.InputDataFields.groundtruth_group_of: (
        #     slim.tfexample_decoder.Tensor('image/object/group_of')),
        # fields.InputDataFields.groundtruth_weights: (
        #     slim.tfexample_decoder.Tensor('image/object/weight')),
    }

    _reader = tf.TFRecordReader
    _decoder = slim.tfexample_decoder.TFExampleDecoder(
        _keys_to_features, _items_to_handlers)

    _dataSet = slim.dataset.Dataset(
        data_sources=FILE_PATTERN,
        reader=_reader,
        decoder=_decoder,
        num_samples=NUMBER_SAMPLE,
        items_to_descriptions={},
        num_classes=NUMBER_CLASSES)

    #run
    with tf.Session() as sess:
        provider = slim.dataset_data_provider.DatasetDataProvider(_dataSet, shuffle=False)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for n in range(_dataSet.num_samples):
            [image, box_coor, box_name] = provider.get(['image', 'groundtruth_boxes','groundtruth_classes',])
            rgbIm = tf.image.convert_image_dtype(image, dtype=tf.uint8)
            rgbIm, box_coor, box_name = sess.run([rgbIm, box_coor, box_name])

            rgbIm = cv2.resize(rgbIm,(448,448))
            h, w, c = np.shape(rgbIm)
            box_coor = box_coor * np.array([w,h,w,h])
            if len(box_coor) != 0:
                for i in range(np.shape(box_coor)[0]):
                    xmin = np.int(box_coor[i][0])
                    ymin = np.int(box_coor[i][1])
                    xmax = np.int(box_coor[i][2])
                    ymax = np.int(box_coor[i][3])
                    cv2.rectangle(rgbIm, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                    txt_pos = [xmax-xmin,ymin]
                    # cv2.putText(rgbIm,'%d' %box_name[i],(np.int(xmin+0.5*txt_pos[0]),np.int(txt_pos[1])),
                    #             cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))

            bgrIm = cv2.cvtColor(rgbIm, cv2.COLOR_RGB2BGR)
            cv2.imshow('verify', bgrIm)
            cv2.waitKey(0)


if __name__ == '__main__':
    verify('val')