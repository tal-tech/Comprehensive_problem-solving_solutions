import tensorflow.contrib.slim as slim
import tensorflow as tf

def base_conv_layer(inputs, widths, is_training):
    batch_norm_params = {'is_training': is_training,
                         'decay': 0.9,
                         'updates_collections': None}

    with slim.arg_scope([slim.conv2d],
                        kernel_size=[3, 3],
                        padding='SAME',
                        weights_regularizer=slim.l2_regularizer(1e-4),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.max_pool2d],
                            kernel_size=[2, 2],
                            stride=[2, 1],
                            padding='SAME'):
            conv1 = slim.conv2d(inputs, 64, scope='conv1')
            conv2 = slim.conv2d(conv1, 64, scope='conv2')
            poo1 = slim.max_pool2d(conv2, kernel_size=[2, 2], stride=[2, 2], scope='pool1')

            conv3 = slim.conv2d(poo1, 128, scope='conv3')
            conv4 = slim.conv2d(conv3, 128, scope='conv4')
            pool2 = slim.max_pool2d(conv4, scope='pool2')

            conv5 = slim.conv2d(pool2, 256, scope='conv5')
            conv6 = slim.conv2d(conv5, 256, scope='conv6')
            pool3 = slim.max_pool2d(conv6, scope='pool3')

            conv7 = slim.conv2d(pool3, 512, scope='conv7')
            conv8 = slim.conv2d(conv7, 512, scope='conv8')

            pool4 = slim.conv2d(conv8, 512, kernel_size=[4, 3], scope='pool4')
            pool5 = slim.max_pool2d(pool4, scope='pool5')

            feat = tf.transpose(pool5, perm=[0, 2, 1, 3], name='pool4_tran')
            feat = tf.reshape(feat, [tf.shape(feat)[0], -1, feat.get_shape()[-1]])

            after_conv1 = widths
            after_pool1 = tf.floor_div(after_conv1, 2)
            after_pool2 = after_pool1
            after_pool3 = after_pool2
            after_pool4 = after_pool3

            sequence_length = tf.reshape(after_pool4, [-1], name='seq_len')
            sequence_length = tf.maximum(2 * sequence_length, 1)

            return feat, sequence_length


def rnn_layer(bottom_sequence, sequence_length, rnn_size, scope):

    cell_fw = tf.contrib.rnn.LSTMBlockCell(rnn_size)
    cell_bw = tf.contrib.rnn.LSTMBlockCell(rnn_size)

    rnn_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, bottom_sequence,
        sequence_length=sequence_length,
        time_major=True,
        dtype=tf.float32,
        scope=scope)

    rnn_output_stack = tf.concat(rnn_output, 2, name='output_stack')

    return rnn_output_stack,enc_state


def rnn_layers(features, sequence_length, num_classes,units):

    logit_activation = tf.nn.relu
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope("rnn"):

        rnn_sequence = tf.transpose(features, perm=[1, 0, 2], name='time_major')
        rnn1 ,_ = rnn_layer(rnn_sequence, sequence_length, units, 'bdrnn1')
        rnn2 ,_ = rnn_layer(rnn1, sequence_length, units, 'bdrnn2')
        rnn_logits = tf.layers.dense(rnn2, num_classes + 1,
                                    activation=logit_activation,
                                    kernel_initializer=weight_initializer,
                                    bias_initializer=bias_initializer,
                                    name='logits')

        return rnn_logits

