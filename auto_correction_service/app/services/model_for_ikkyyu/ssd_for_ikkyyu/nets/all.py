# -*- coding:UTF-8 -*-
import math
import numpy as np
import tensorflow as tf
import tf_extended as tfe
from collections import namedtuple
from nets import custom_layers
from nets import ssd_common
from tensorflow.contrib import slim
from tensorflow import keras

# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])

filter_size = {
    2: [3,5],
    3: [3,5],
    5: [3,7],
    7: [3,9],
    10: [3,13],
    15: [3,17] ,
    .5:[5,3],
    .33:[5,3]
}
class SSDNet(object):
    """Implementation of the SSD VGG-based 512 network.

    The default features layers with 512x512 image input are:
      conv4 ==> 64 x 64
      conv7 ==> 32 x 32
      conv8 ==> 16 x 16
      conv9 ==> 8 x 8
      conv10 ==> 4 x 4
      conv11 ==> 2 x 2
      conv12 ==> 1 x 1
    The default image size used to train this network is 512x512.
    """
    default_params = SSDParams(
        img_shape=(512, 512),
        num_classes=4,
        no_annotation_label=1,
        feat_layers=['block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9'],
        feat_shapes=[(128, 128), (64, 64), (32, 32), (16, 16), (8, 8), (8, 8), (8, 8)],
        anchor_size_bounds=[0.06, 0.5],
        anchor_sizes=[(12, 25),
                      (25, 50),
                      (50, 100),
                      (100, 200),
                      (200, 300),
                      (300, 400),
                      (400, 500)],
        anchor_ratios=[[2, .5, 3, .33, 5, 7, 10, 15],
                       [2, .5, 3, .33, 5, 7, 10, 15],
                       [2, .5, 3, .33, 5, 7, 10, 15],
                       [2, .5, 3, .33, 5, 7, 10, 15],
                       [2, .5, 3, .33, 5],
                       [2, .5, 3, .33, 5],
                       [2, .5, 3, .33, 5]],
        anchor_steps=[4, 8, 16, 32, 64, 64, 64],
        anchor_offset= [[0.25, 0.5, 0.75],
                        [0.333, 0.666],
                        [0.333, 0.666],
                        [0.333, 0.666],
                        [0.5],
                        [0.5],
                        [0.5],
                       ],
        normalizations=[20, -1, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_512_vgg'):
        """Network definition.
        """
        
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    is_training=is_training,
                    prediction_fn=prediction_fn,
                    reuse=reuse)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #
    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)
    
    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        # if clipping_bbox is not None:
        #     rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,loss_type, batch_size,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)

def tensor_shape(x, rank=3):
    """
    Args:
        image: A N-D Tensor of shape.
    Returns:
        A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape  = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]
# =========================================================================== #
# SSD tools...
# =========================================================================== #
def layer_shape(layer):
    """Returns the dimensions of a 4D layer tensor.
    Args:
      layer: A 4-D Tensor of shape `[height, width, channels]`.
    Returns:
      Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if layer.get_shape().is_fully_defined():
        return layer.get_shape().as_list()
    else:
        static_shape = layer.get_shape().with_rank(4).as_list()
        dynamic_shape = tf.unstack(tf.shape(layer), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(512, 512)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (512 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * 0.04, img_size * 0.1]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes



def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        shape = l.get_shape().as_list()[1:4]
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes



def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset = 0.5,
                         index = 0,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + 0.5) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    
    if index!= 0: #kk:每一层都有俩ratio<1的
        h = np.zeros((num_anchors-2, ), dtype=dtype)
        w = np.zeros((num_anchors-2, ), dtype=dtype)
    else:
        h = np.zeros((num_anchors, ), dtype=dtype)
        w = np.zeros((num_anchors, ), dtype=dtype)
        
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        if not (r < 1 and index != 0):
#             if offset != offset_list[(len(offset_list)-1)//2]:
            h[di] = sizes[0] / img_shape[0] / math.sqrt(r)
            w[di] = sizes[0] / img_shape[1] * math.sqrt(r)
            di += 1
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset_list = [0.5],
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        for index, offset in enumerate(offset_list[i]):
            anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                                 anchor_sizes[i],
                                                 anchor_ratios[i],
                                                 anchor_steps[i],
                                                 offset = offset, 
                                                 index = index,
                                                 dtype = dtype)
            layers_anchors.append(anchor_bboxes)
    return layers_anchors

def conv_transpose_bn_relu(x, out_channel):
    with tf.variable_scope(None, 'conv_tran_bn_relu'):
        x = slim.conv2d_transpose(x, out_channel, [4, 4], stride=[2, 2])
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
    return x


def conv_bn_relu(x, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn_relu'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
    return x


def fpn_layer(inputs, out_channel, is_training=True):

    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with tf.variable_scope(None, 'AddFpnLayer'):
            inputs.reverse()
            
            _p0 = conv_bn_relu(inputs[0], out_channel, [3, 3], 1)
            _p1 = conv_bn_relu(inputs[1], out_channel, [3, 3], 1)
            _p2 = conv_bn_relu(inputs[2], out_channel, [3, 3], 1)

            # [batch 8 8 976]->[batch 16 16 64]
            _s0 = conv_bn_relu(_p0, out_channel, [3, 3], 1)
            _t0 = conv_transpose_bn_relu(_s0, out_channel)
            _l0 = conv_bn_relu(inputs[3], out_channel, 1, 1)
            _p3 = keras.layers.Concatenate(axis=-1)([_t0, _l0])

            # [batch 16 16 448]->[batch 32 32 64] #18M
            _s1 = conv_bn_relu(_p3, out_channel, [3, 3], 1)
            _t1 = conv_transpose_bn_relu(_s1, out_channel)
            _l1 = conv_bn_relu(inputs[4], out_channel, 1, 1)
            _p4 = keras.layers.Concatenate(axis=-1)([_t1, _l1])

            # [batch 32 32 116]->[batch 64 64 64] #144M
            _s2 = conv_bn_relu(_p4, out_channel, [3, 3], 1)
            _t2 = conv_transpose_bn_relu(_s2, out_channel)
            _l2 = conv_bn_relu(inputs[5], out_channel, 1, 1)
            _p5 = keras.layers.Concatenate(axis=-1)([_t2, _l2])
            
            _s3 = conv_bn_relu(_p5, out_channel, [3, 3], 1)
            _t3 = conv_transpose_bn_relu(_s3, out_channel)
            _l3 = conv_bn_relu(inputs[6], out_channel, 1, 1)
            _p6 = keras.layers.Concatenate(axis=-1)([_t3, _l3])
            return [_p6, _p5, _p4, _p3, _p2, _p1, _p0]

        



def fcn_layer(inputs,
              out_channels_for_fcn,
              out_channels,
              filter_size = [3,3],
              is_training=True,
              scope='AddFcnLayer'):
    """
    full convolution network
    """
    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with tf.variable_scope(None, scope):
            out = conv_bn_relu(inputs, out_channels_for_fcn, filter_size, 1)
            out = conv_bn_relu(out, out_channels_for_fcn, filter_size, 1)
            out = slim.conv2d(out, out_channels, kernel_size=[1, 1], activation_fn=None)
            return out


        

# =========================================================================== #
# Functional definition of VGG-based SSD 512.
# =========================================================================== #

def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            anchor_offset = SSDNet.default_params.anchor_offset,
            is_training=True,
            prediction_fn=slim.softmax,
            reuse=None,):
    """SSD net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope('vgg_16', 'vgg_16', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net #128
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net #64
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net #32
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool5')
        
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with tf.variable_scope(None, 'new_vgg'):
                # Block 6.
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv6')
                end_points['block6'] = net #16 1:32
                net = slim.max_pool2d(net, [2, 2], 2, scope='pool6')

                # Block 7.
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv7')
                end_points['block7'] = net #8 1:64

                net = slim.conv2d(net, 1024, [3, 3], rate=2, scope='conv8')
                end_points['block8'] = net #8 1:64

                net = slim.conv2d(net, 1024, [3, 3], rate=2, scope='conv9')
                end_points['block9'] = net #8 1:64
        

        ssd_out = []
        for i, layer in enumerate(feat_layers):
            ssd_out.append(end_points[layer])
        fpn_out = fpn_layer(ssd_out, 64, is_training=is_training)

        predictions, logits, localisations = [], [], []
        for i, lay in enumerate(fpn_out):
            for offset_index, offset in enumerate(anchor_offset[i]):
                num_anchors = len(anchor_sizes[i]) + len(anchor_ratios[i])
                single_channel = (num_classes + 4)
                fcn = fcn_layer(lay, 64, single_channel, is_training=is_training, scope='AddFcnLayer_%d/0' % (i))
                loc = fcn[:, :, :, :4]
                cls = fcn[:, :, :, 4:]

                fcn = fcn_layer(lay, 64, single_channel, is_training=is_training, scope='AddFcnLayer_%d/1' % (i))
                loc_temp = fcn[:, :, :, :4]
                cls_temp = fcn[:, :, :, 4:]
                loc = keras.layers.Concatenate(axis=-1)([loc, loc_temp])
                cls = keras.layers.Concatenate(axis=-1)([cls, cls_temp])


                for j in range(2, num_anchors):
                    if offset_index == 0 or anchor_ratios[i][j-2] > 1: #对于anchor_ratio<1的anchor,不需要添加offset
                        fcn = fcn_layer(lay, 64, single_channel, filter_size[anchor_ratios[i][j-2]], is_training=is_training, scope='AddFcnLayer_%d/%d' % (i, j))
                        loc_temp = fcn[:, :, :, :4]
                        cls_temp = fcn[:, :, :, 4:]
                        loc = keras.layers.Concatenate(axis=-1)([loc, loc_temp])
                        cls = keras.layers.Concatenate(axis=-1)([cls, cls_temp])
                        
                if offset_index != 0:
                    num_anchors -= 2
                loc = tf.reshape(loc, tensor_shape(loc, 4)[:-1] + [num_anchors, 4])
                cls = tf.reshape(cls, tensor_shape(cls, 4)[:-1] + [num_anchors, num_classes])
                logits.append(cls)
                localisations.append(loc)
                predictions.append(prediction_fn(cls))
       


        return predictions, localisations, logits, end_points
    
ssd_net.default_image_size = 512



def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #

def ssd_arg_scope_caffe(caffe_scope):
    """Caffe scope definition.

    Args:
      caffe_scope: Caffe scope object with loaded weights.

    Returns:
      An arg_scope.
    """
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc

def focal_loss(x, y, num_classes):
    alpha = 0.25
    gamma = 2

    y = tf.cast(y, tf.int32)
    t = tf.one_hot(y, depth=num_classes+1)  # [N, #total_cls]      #num_classes + 1

    p = tf.sigmoid(x)
    pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
    w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
    w = w * tf.pow((1 - pt), gamma)

    loss = tf.losses.sigmoid_cross_entropy(t, x, w)

    return loss

# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def ssd_losses(logits, localisations, gclasses, glocalisations, gscores,
               match_threshold=0.5, negative_ratio=3., alpha=1., label_smoothing=0.,
               scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tfe.get_shape(logits[0], 5)
        num_classes = lshape[-1]

        pre_loc_l, g_loc_l, g_cls_l, pre_cls_l = [], [], [], []
        flogits, fgclasses, fgscores, flocalisations, fglocalisations = [], [], [], [], []
        # print(logits[0].shape) # 4*128*128*8*4
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))

        for i in range(len(flogits)):
            dtype = flogits[i].dtype

            with tf.name_scope('block_%i' % i):
                # Determine weights Tensor.
                pmask = fgscores[i] > match_threshold
                fpmask = tf.cast(pmask, dtype)
                ipmask = tf.cast(pmask, tf.int64)
                n_positives = tf.reduce_sum(fpmask)

                nmask = tf.logical_and(tf.logical_not(pmask), fgscores[i] > -0.5)
#                 nmask = tf.logical_and(tf.logical_not(pmask), fgscores[i] > 0.1)
                fnmask = tf.cast(nmask, dtype)
                print(flogits[i].shape)
                predictions = slim.softmax(flogits[i])
                nvalues = tf.where(nmask, predictions[:, 0], 1. - fnmask)
                nvalues_flat = tf.reshape(nvalues, [-1])

                # Number of negative entries to select.
                max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                n_neg = tf.minimum(n_neg, max_neg_entries) + 1

                val, idxes = tf.nn.top_k(-nvalues_flat, k = n_neg)
                max_hard_pred = -val[-1]
                nmask = tf.logical_and(nmask, nvalues < max_hard_pred)

                # mask for localization
                pre_loc = tf.boolean_mask(flocalisations[i], pmask)
                g_loc = tf.boolean_mask(fglocalisations[i], pmask)

                # mask for classification
                np_mask = tf.logical_or(nmask, pmask)
                g_cls = tf.boolean_mask(ipmask * fgclasses[i], np_mask)
#                 flogits[np.array(flogits) < 1e-10 and np.array(flogits) > -(1e-10)] = 1e-10
#                 flogits[np.array(flogits) > (1 -(1e-10)) and np.array(flogits) < (1 + (1e-10))] = 1 - (1e-10)
                pre_cls = tf.boolean_mask(flogits[i], np_mask)

                # store
                pre_cls_l.append(pre_cls)
                g_cls_l.append(g_cls)
                pre_loc_l.append(pre_loc)
                g_loc_l.append(g_loc)

        # cal localization loss and classification loss
        p_cls = tf.concat(pre_cls_l, axis=0)
        g_cls = tf.concat(g_cls_l, axis=0)
        p_loc = tf.concat(pre_loc_l, axis=0)
        g_loc = tf.concat(g_loc_l, axis=0)

        with tf.name_scope('cross_entropy'):
            loss = tf.losses.sparse_softmax_cross_entropy(logits=p_cls, labels=g_cls)
#             loss = focal_loss(p_cls, g_cls, 3)
            tf.losses.add_loss(1.0 * loss)

        with tf.name_scope('localization'):
            loss = custom_layers.l1(p_loc, g_loc)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
            tf.losses.add_loss(1.0 * loss)
