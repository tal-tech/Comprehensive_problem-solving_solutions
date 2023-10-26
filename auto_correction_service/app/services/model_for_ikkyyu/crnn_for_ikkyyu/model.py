import tensorflow as tf
from recognition_config import Config as config
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
from recognition_utils import image_normalization
from recognition_inference import base_conv_layer, rnn_layer, rnn_layers
import os
import ocr_beam.det_beam_search as det_beam_search
import random
from tqdm import tqdm
from math import log
from Dataset import DataSet
os.environ['CUDA_VISIBLE_DEVICES']='0'


class CTC_Model(object):

    def __init__(self, model_name='crnn', is_training=False, model_path=None):
        if is_training:
            self.name = model_name
            self.train_board_path = './' + model_name + '_train_board'
            self.val_board_path = './' + model_name + '_val_board'
            if not os.path.exists(self.name):
                os.mkdir(self.name)
        # 初始化图结构
        self.config = config()
        self.g = tf.Graph()

        # 添加对显存的限制 add by boby 201905
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.sess = tf.Session(config=gpu_config, graph=self.g)

        # self.sess = tf.Session(graph=self.g)
        
        with self.sess.as_default():
            with self.g.as_default():
                self.inputs = tf.placeholder(tf.float32, [None,32, None, 1])
                self. width = tf.placeholder(tf.int32, [None])
                self.is_training = tf.placeholder(tf.bool)
                self.logits, self.sequence_length = self.crnn(self.inputs, self.width, self.is_training)
                self.decodes_greedy, self.decodes_greedy_prob = tf.nn.ctc_greedy_decoder(self.logits, self.sequence_length, merge_repeated=True)
                self.decodes_greedy = self.decodes_greedy[0]
                self.dense_greedy_decoder = tf.sparse_to_dense(
                    sparse_indices=self.decodes_greedy.indices, output_shape=self.decodes_greedy.dense_shape, 
                    sparse_values=self.decodes_greedy.values, default_value=-1
                )
                if not is_training:
                    saver = tf.train.Saver(tf.global_variables())
                    if not model_path:
                        if os.path.exists(self.config.MODEL_SAVE + '.meta'):
                            saver.restore(self.sess, self.config.MODEL_SAVE)
                        else:
                            raise Exception("请检查ckpt文件路径配置")
                    else:
                        if os.path.exists(model_path + '.meta'):
                            saver.restore(self.sess, model_path)
                        else:
                            raise Exception("请检查ckpt文件路径配置")

    def crnn(self,inputs, width,is_training):

        features, sequence_length = base_conv_layer(inputs, width,is_training)
        logits = rnn_layers(features, sequence_length, len(self.config.ONE_HOT),config.RNN_UNITS)
        return logits ,sequence_length
    
    @staticmethod
    def ctc_loss_layer(rnn_logits, sequence_labels, sequence_length):
        """Build CTC Loss layer for training"""
        loss = tf.nn.ctc_loss(sequence_labels, rnn_logits, sequence_length,
                              time_major=True, ignore_longer_outputs_than_inputs=True)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss)
        return loss

    @staticmethod
    def compute_acc(label_errors, label_length, name):

        sequence_errors = tf.count_nonzero(label_errors, axis=0)

        total_label_error = tf.reduce_sum(label_errors)
        total_labels = tf.reduce_sum(label_length)
        label_error = tf.truediv(total_label_error,
                                 tf.cast(total_labels, tf.float32),
                                 name=name + '_' + 'label_error')

        sequence_error = tf.truediv(tf.cast(sequence_errors, tf.int32),
                                    tf.shape(label_length)[0],
                                    name=name + '_' + 'sequence_error')

        label_acc = 1 - label_error
        seq_acc = 1 - sequence_error

        tf.summary.scalar(name + '_' + 'label_acc', label_acc)
        tf.summary.scalar(name + '_' + 'seq_acc', seq_acc)

        return label_acc, seq_acc

    @staticmethod
    def error(logits, sequence_length, sequence_label, label_length, greedy_decoder=True):

        '''
        计算各个评价指标，
        :param logits: 逻辑回归的值
        :param sequence_length:
        :param sequence_label:
        :param label_length:
        :param greedy_decoder: 是否使用贪心编码
        :return:
        '''

        if greedy_decoder:
            predictions, _ = tf.nn.ctc_greedy_decoder(logits, sequence_length, merge_repeated=True)
        else:
            predictions, _ = tf.nn.ctc_beam_search_decoder(logits, sequence_length, beam_width=10, top_paths=5,
                                                           merge_repeated=False)

        decoder = predictions[0]
        dense_decoder = tf.sparse_to_dense(sparse_indices=decoder.indices, output_shape=decoder.dense_shape,
                                           sparse_values=decoder.values, default_value=-1)

        hypothesis = tf.cast(predictions[0], tf.int32)  # for edit_distance

        label_errors = tf.edit_distance(hypothesis, sequence_label, normalize=False)

        sequence_errors = tf.count_nonzero(label_errors, axis=0)

        total_label_error = tf.reduce_sum(label_errors)
        total_labels = tf.reduce_sum(label_length)
        label_error = tf.truediv(total_label_error,
                                 tf.cast(total_labels, tf.float32),
                                 name='label_error')

        sequence_error = tf.truediv(tf.cast(sequence_errors, tf.int32),
                                    tf.shape(label_length)[0],
                                    name='sequence_error')

        label_acc = 1 - label_error
        seq_acc = 1 - sequence_error

        return label_acc, seq_acc, dense_decoder


    def display_result(self, train_data, val_data, epoch):
        decode = dict(zip(self.config.ONE_HOT.values(), self.config.ONE_HOT.keys()))
        train_loss = 'train_loss: all:{} '.format(train_data['all_loss'])
        train_label_acc = 'train_label_acc: all:{} '.format(train_data['all_char_acc'])
        train_seq_acc = 'train_seq_acc : all:{}'.format(train_data['all_seq_acc'])
        val_loss = 'val_loss: all:{} '.format(val_data['all_loss'])
        val_label_acc = 'val_label_acc: all:{} '.format(val_data['all_char_acc'])
        val_seq_acc = 'val_seq_acc : all:{}'.format(val_data['all_seq_acc'])

        train_real_labels = []
        val_real_labels = []
        train_pre = []
        val_pre = []

        for i, name in enumerate(self.config.TRAIN_DATA_NAME):
            train_loss = train_loss + '; ' + name + ':' + str(train_data[name + '_' + 'loss'])
            train_label_acc = train_label_acc + '; ' + name + ':' + str(train_data[name + '_' + 'char_acc'])
            train_seq_acc = train_seq_acc + '; ' + name + ':' + str(train_data[name + '_' + 'seq_acc'])
            train_dense_decoder = train_data[name + '_' + 'dense_decoder'].tolist()
            train_result = list(
                map(lambda y: ''.join(list(map(lambda x: decode.get(x), y))), train_dense_decoder))
            num = random.randint(0, len(train_result) - 1)
            train_real_labels.append(train_data[name + '_' + 'real_labels'][num])
            train_pre.append(train_result[num])

        for i, name in enumerate(config.VAL_DATA_NAME):
            val_loss = val_loss + '; ' + name + ':' + str(val_data[name + '_' + 'loss'])
            val_label_acc = val_label_acc + '; ' + name + ':' + str(val_data[name + '_' + 'char_acc'])
            val_seq_acc = val_seq_acc + '; ' + name + ':' + str(val_data[name + '_' + 'seq_acc'])
            val_dense_decoder = val_data[name + '_' + 'dense_decoder'].tolist()
            val_result = list(
                map(lambda y: ''.join(list(map(lambda x: decode.get(x), y))), val_dense_decoder))
            num = random.randint(0, len(train_result) - 1)
            val_real_labels.append(val_data[name + '_' + 'real_labels'][num])
            val_pre.append(val_result[num])

        print(train_loss)
        print(train_label_acc)
        print(train_seq_acc)
        print(val_loss)
        print(val_label_acc)
        print(val_seq_acc)
        print('epoch{}'.format(epoch))
        print('train_label{}'.format(train_real_labels))
        print('train_result{}'.format(train_pre))
        print('val_label{}'.format(val_real_labels))
        print('val_result{}'.format(val_pre))
        print('-------------------------------------------------------------------------------------------------------')

    def train(self):

        def add_summery(name, all_inputs, data_list, summer_writer):
            summary = tf.Summary()
            images_train, labels_train, width_train, length_train, real_labels_train = all_inputs
            feeddict_train = {self.inputs: images_train, self.sequence_label: (labels_train[0], labels_train[1], labels_train[2]),
                              self.width: width_train, self.label_length: length_train, self.is_training: False}
            label_acc_, seq_acc_, dense_decoder_, loss_ = self.sess.run([self.label_acc, self.seq_acc, self.dense_decoder, self.loss],
                                                                   feed_dict=feeddict_train)

            data_list[name + '_' + 'loss'] = loss_
            data_list[name + '_' + 'char_acc'] = label_acc_
            data_list[name + '_' + 'seq_acc'] = seq_acc_
            data_list[name + '_' + 'dense_decoder'] = dense_decoder_
            data_list[name + '_' + 'real_labels'] = real_labels_train
            summary.value.add(tag=name + '/' + 'loss', simple_value=label_acc_)
            summary.value.add(tag=name + '/' + 'char_acc', simple_value=label_acc_)
            summary.value.add(tag=name + '/' + 'seq_acc', simple_value=seq_acc_)
            summer_writer.add_summary(summary, step)
        with self.sess.as_default():
            with self.g.as_default():
                self.sequence_label = tf.sparse_placeholder(tf.int32)
                self.label_length = tf.placeholder(tf.int32, [None])
                self.loss = self.ctc_loss_layer(self.logits, self.sequence_label, self.sequence_length)
                self.label_acc, self.seq_acc, self.dense_decoder = self.error(self.logits, self.sequence_length, self.sequence_label, self.label_length)
                optimizer = tf.train.AdamOptimizer(self.config.LEARN_RATE).minimize(self.loss)
                saver = tf.train.Saver(tf.global_variables())
                if os.path.exists(self.config.MODEL_SAVE + '.meta'):
                    saver.restore(self.sess, self.config.MODEL_SAVE)
                    step = self.config.MODEL_SAVE.split('/')[-1].split('.')[0]
                    step = int(step)
                    print("restore,setp:{}".format(step))
                else:
                    step = 0
                    self.sess.run(tf.global_variables_initializer())

                dataset = DataSet()
                train_generator = dataset.trian_generator()
                display_generator = dataset.display_generator()
                all_val_data = dataset.create_val_data()
                merged = tf.summary.merge_all()
                writer_train = tf.summary.FileWriter(self.train_board_path, self.g)
                writer_val = tf.summary.FileWriter(self.val_board_path, self.g)
                seq_acc_max = [-1] * len(config.TRAIN_DATA_NAME)  # 每个数据集最大的seq_acc
                all_sequence_acc_equ_max = -1  # 所有数据最大的seq_acc
                while True:
                    images, labels, width_, length_, real_labels, epoch = next(train_generator)
                    feeddict = {self.inputs: images, self.sequence_label: (labels[0], labels[1], labels[2]), self.width: width_,
                                self.label_length: length_, self.is_training: True}
                    self.sess.run(optimizer, feed_dict=feeddict)


                    if (step + 0) % 500 == 0:
                        train_inputs, val_inputs = next(display_generator)
                        train_data = {}
                        val_data = {}
                        images_train, labels_train, width_train, length_train, real_labels_train = train_inputs[-1]
                        feeddict_train = {self.inputs: images_train, self.sequence_label: (labels_train[0], labels_train[1], labels_train[2]),
                                            self.width: width_train, self.label_length: length_train, self.is_training: False}
                        merged_train = self.sess.run(merged, feed_dict=feeddict_train)
                        writer_train.add_summary(merged_train)
                        for i, train_name in enumerate(config.TRAIN_DATA_NAME):
                            add_summery(train_name, train_inputs[i], train_data, writer_train)
                        add_summery('all', train_inputs[-1], train_data, writer_train)
                        for i, val_name in enumerate(config.VAL_DATA_NAME):
                            add_summery(val_name, val_inputs[i], val_data, writer_val)
                        add_summery('all', val_inputs[-1], val_data, writer_val)
                        self.display_result(train_data, val_data, epoch)
                    if (step + 1) % 2000 == 0:
                        print('calculating precision rate for each dataset')
                        label_acc_list = []  # 每一个数据集的label_acc
                        seq_acc_list = []  # 每一个数据集的seq_acc
                        label_acc_string = 'val_label_acc_all:'
                        seq_acc_string = 'val_seq_acc_all:'
                        summary = tf.Summary()

                        for num, all_data in enumerate(all_val_data):
                            label_acc_all, sequence_acc_all = 0, 0
                            j = 0
                            print('calculating{}'.format(config.VAL_DATA_NAME[num]))
                            for i in tqdm(range(len(all_data))):
                                images_val, labels_val, width_val, length_val, _, _ = all_data[i]
                                feeddict_val = {self.inputs: images_val,
                                                self.sequence_label: (labels_val[0], labels_val[1], labels_val[2]),
                                                self.width: width_val, self.label_length: length_val, self.is_training: False}

                                label_acc_, seq_acc_ = self.sess.run([self.label_acc, self.seq_acc], feed_dict=feeddict_val)
                                label_acc_all = label_acc_all + label_acc_
                                sequence_acc_all = sequence_acc_all + seq_acc_
                                j = j + 1

                            label_acc_equ = label_acc_all / j
                            sequence_acc_equ = sequence_acc_all / j
                            summary.value.add(tag='Vaild/' + config.VAL_DATA_NAME[num] + '_' + 'seq_acc',
                                                simple_value=sequence_acc_equ)
                            label_acc_list.append(label_acc_equ)
                            seq_acc_list.append(sequence_acc_equ)
                            label_acc_string = label_acc_string + ' ' + config.VAL_DATA_NAME[num] + ':' + str(label_acc_equ)
                            seq_acc_string = seq_acc_string + ' ' + config.VAL_DATA_NAME[num] + ':' + str(sequence_acc_equ)

                            # 每个数据集的准确度提高，存储模型
                            if seq_acc_max[num] < sequence_acc_equ:
                                seq_acc_max[num] = sequence_acc_equ
                                path = './' + self.name + '/' + config.VAL_DATA_NAME[num] + '/' + str(step) + '.ckpt'
                                saver.save(self.sess, path)

                        all_label_acc_equ = np.asarray(label_acc_list, dtype=np.float32)
                        all_label_acc_equ = np.mean(all_label_acc_equ)
                        all_sequence_acc_equ = np.asarray(seq_acc_list, dtype=np.float32)
                        all_sequence_acc_equ = np.mean(all_sequence_acc_equ)
                        summary.value.add(tag='Vaild/' + 'alldata' + '_' + 'seq_acc',
                                            simple_value=all_sequence_acc_equ)

                        if all_sequence_acc_equ_max < all_sequence_acc_equ:
                            all_sequence_acc_equ_max = all_sequence_acc_equ
                            path = './' + self.name + '/' + 'alldata' + '/' + str(step) + '.ckpt'
                            saver.save(self.sess, path)
                        writer_val.add_summary(summary, step)

                        label_acc_string = label_acc_string + ' ' + 'average' + ':' + str(all_label_acc_equ)
                        seq_acc_string = seq_acc_string + ' ' + 'average' + ':' + str(all_sequence_acc_equ)
                        f = open('log.txt', 'a', encoding='utf-8')
                        f.write(label_acc_string)
                        f.write(seq_acc_string)
                        f.write('-----------------------------------------------------------------------------------------')
                        print(label_acc_string)
                        print(seq_acc_string)
                        print('-------------------------------------------------------------------------------------------')

                    step = step + 1
            
    def greedy_search(self, images, widths):
        def sentence_to_output(sentence):
            output = sentence.tolist()
            decode = dict(zip(self.config.ONE_HOT.values(), self.config.ONE_HOT.keys()))
            output = list(map(lambda y: ''.join(list(map(lambda x: decode.get(x), y))).replace('－', '-'), output))
            return output
        logits, decodes_greedy, sequence_length = self.sess.run([self.logits, self.dense_greedy_decoder, self.sequence_length], 
        feed_dict={self.inputs: images, self.width: widths, self.is_training: False})
        output_list = sentence_to_output(decodes_greedy)
        logits = logits[..., self.config.NUM_SIGN]
        logits = np.swapaxes(logits, 0, 1)
        return output_list, logits, sequence_length

    def beam_search(self, logits, sequence_length, beamsearch_width=5):
        det_beam_search.Init(beamsearch_width, 45, 8)
        float_vec_vec_vec = det_beam_search.float_vec_vec_vec()
        for n in range(logits.shape[0]):
            float_vec_vec = det_beam_search.float_vec_vec()
            for sl in range(sequence_length[n]):
                float_vec = det_beam_search.float_vec()
                for col in range(logits.shape[2]):
                    float_vec.append(float(logits[n, sl, col]))
                float_vec_vec.append(float_vec)
            float_vec_vec_vec.append(float_vec_vec)
        result_vec_vec_vec = det_beam_search.int_vec_vec_vec()
        conf_vec = det_beam_search.float_vec_vec()
        det_beam_search.detection_vec_thread(float_vec_vec_vec, result_vec_vec_vec, conf_vec)
        output_list = []
        for oneResult in result_vec_vec_vec:
            result_list = []
            for result in oneResult:
                result = ''.join(map(lambda x: config.DECODE[x], result))
                result_list.append(result)
            output_list.append(result_list)
        return output_list, conf_vec

    def beam_search_interface(self, images, widths):
        """
        对外封装了一层接口
        20190516 add by boby
        :param images:
        :param widths:
        :return:
        """
        logits, sequence_length = self.sess.run([self.logits, self.sequence_length], feed_dict={self.inputs: images, self.width: widths, self.is_training: False})
        logits = logits[..., self.config.NUM_SIGN]
        logits = np.swapaxes(logits, 0, 1) # 轴的交换
        output_list, conf_vec = self.beam_search(logits, sequence_length)
        return output_list, conf_vec


    def tf_beam_search(self, logits, sequence_length):
        def sentence_to_output(sentence):
            output = sentence[0].tolist()
            output = ''.join([self.config.DECODE[i] for i in output])
            return output
        g2 = tf.Graph()
        sess2 = tf.Session(graph=g2)
        with sess2.as_default():
            with g2.as_default():
                logits = np.transpose(logits, (1, 0, 2))
                logits = tf.convert_to_tensor(logits)
                predictions, _ = tf.nn.ctc_beam_search_decoder(logits, sequence_length, beam_width=40, top_paths=5, merge_repeated=False)
                dense_beam_decoders = list()
                for prediction in predictions:
                    dense_beam_decoders.append(tf.sparse_to_dense(
                        sparse_indices=prediction.indices, output_shape=prediction.dense_shape, 
                        sparse_values=prediction .values, default_value=-1
                    ))
        
        start = time.time()
        decodes_beam_search = sess2.run(dense_beam_decoders)
        output_list = [sentence_to_output(i) for i in decodes_beam_search]
        end = time.time()
        print('ctc{}s'.format(end-start))
        return output_list

    
if __name__ == "__main__":
    import time
    # img = cv2.imread("/home/yichao/Pictures/66.png")
    model = CTC_Model(is_training=True, model_name='crnn')
    model.train()
    # image = np.array(image_normalization(img))
    # widths = np.array([image.shape[1]])
    # image = image[np.newaxis, ..., np.newaxis]  
    # output_list, logits, sequence_length = model.greedy_search(image, widths)

    # print(model.tf_beam_search(logits, sequence_length))
    # start = time.time()
    # print(model.beam_search(logits, sequence_length))
    # end = time.time()
    # print(end - start)
