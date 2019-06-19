from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf

from tensorflow.contrib import keras
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import variable_scope

class SegDecModel(object):

    # If a model is trained using multiple GPUs, prefix all Op names with tower_name
    # to differentiate the operations. Note that this prefix is removed from the
    # names of the summaries when visualizing a model.
    TOWER_NAME = 'tower'

    # Batch normalization. Constant governing the exponential moving average of
    # the 'global' mean and variance for all activations.
    BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

    # The decay to use for the moving average.
    MOVING_AVERAGE_DECAY = 0.9999

    DECISION_NET_NONE = 0
    DECISION_NET_LOGISTIC = 1
    DECISION_NET_FULL = 2

    def __init__(self,
                 use_corss_entropy_seg_net=True,
                 positive_weight=1,
                 decision_net=DECISION_NET_NONE,
                 decision_positive_weight=1,
                 load_from_seg_only_net=False):

        # weight for positive samples in segmentation net
        self.positive_weight = positive_weight

        # weight for positive samples in decision net
        self.decision_positive_weight = decision_positive_weight

        if decision_net == SegDecModel.DECISION_NET_NONE:
            self.decision_net_fn = lambda net, net_prob_mat: None
        elif decision_net == SegDecModel.DECISION_NET_LOGISTIC:
            self.decision_net_fn = self.get_decision_net_simple
        elif decision_net == SegDecModel.DECISION_NET_FULL:
            self.decision_net_fn = self.get_decision_net

        self.use_corss_entropy_seg_net = use_corss_entropy_seg_net

        # this is only when loading from pre-trained network of segmetnation that did not have decision net layers
        # present at the same time
        self.load_from_seg_only_net = load_from_seg_only_net




    def get_inference(self, inputs, num_classes, for_training=False, restore_logits=True, scope=None):
      """ Build model
      
    
      Args:
        images: Images returned from inputs() or distorted_inputs().
        num_classes: number of classes
        for_training: If set to `True`, build the inference model for training.
          Kernels that operate differently for inference during training
          e.g. dropout, are appropriately configured.
        restore_logits: whether or not the logits layers should be restored.
          Useful for fine-tuning a model with different num_classes.
        scope: optional prefix string identifying the ImageNet tower.
    
      Returns:
        Logits. 2-D float Tensor.
        Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
      """


      with variable_scope.variable_scope(scope, 'SegDecNet', [inputs]) as sc:
          end_points_collection = sc.original_name_scope + '_end_points'
          # Collect outputs for conv2d, max_pool2d
          with arg_scope(
                  [layers.conv2d, layers.fully_connected, layers_lib.max_pool2d, layers.batch_norm],
                  outputs_collections=end_points_collection):

              # Apply specific parameters to all conv2d layers (to use batch norm and relu - relu is by default)
              with arg_scope([layers.conv2d, layers.fully_connected],
                             weights_initializer= lambda shape,dtype=tf.float32, partition_info=None: tf.random_normal(shape, mean=0,stddev=0.01, dtype=dtype),
                             biases_initializer=None,
                             normalizer_fn=layers.batch_norm,
                             normalizer_params={'center': True,
                                                'scale': True,
                                                #'is_training': for_training, # we disable this to do feature normalization (but requires batch size=1)
                                                'decay': self.BATCHNORM_MOVING_AVERAGE_DECAY, # Decay for the moving averages.
                                                'epsilon': 0.001, # epsilon to prevent 0s in variance.
                                                }):

                  net = layers_lib.repeat(inputs, 2, layers.conv2d, 32, [5, 5], scope='conv1')

                  net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')

                  net = layers_lib.repeat(net, 3, layers.conv2d, 64, [5, 5], scope='conv2')

                  net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')

                  net = layers_lib.repeat(net, 4, layers.conv2d, 64, [5, 5], scope='conv3')

                  net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')

                  net = layers.conv2d(net, 1024, [15, 15], padding='SAME', scope='conv4')

                  net_prob_mat = layers.conv2d(net, 1, [1, 1], scope='conv5',
                                               activation_fn=None)

                  decision_net = self.decision_net_fn(net, tf.nn.relu(net_prob_mat))

                  # Convert end_points_collection into a end_point dict.
                  endpoints = utils.convert_collection_to_dict(end_points_collection)



      # Add summaries for viewing model statistics on TensorBoard.
      self._activation_summaries(endpoints)

      return net_prob_mat, decision_net, endpoints

    def get_decision_net_simple(self, net, net_prob_mat):

        avg_output = keras.layers.GlobalAveragePooling2D()(net_prob_mat)
        max_output = keras.layers.GlobalMaxPooling2D()(net_prob_mat)

        decision_net = tf.concat([avg_output, max_output], 3)

        decision_net = layers.conv2d(decision_net, 1, [1, 1], scope='decision6',
                                     normalizer_fn=None,
                                     weights_initializer=initializers.xavier_initializer_conv2d(False),
                                     biases_initializer=tf.constant_initializer(0),
                                     activation_fn=None)

        return decision_net

    def get_decision_net(self, net, net_prob_mat):

        with tf.name_scope('decision'):

            decision_net = tf.concat([net, net_prob_mat],axis=3)

            decision_net = layers_lib.max_pool2d(decision_net, [2, 2], scope='decision/pool4')

            decision_net = layers.conv2d(decision_net, 8, [5, 5], padding='SAME', scope='decision/conv6')

            decision_net = layers_lib.max_pool2d(decision_net, [2, 2], scope='decision/pool5')

            decision_net = layers.conv2d(decision_net, 16, [5, 5], padding='SAME', scope='decision/conv7')

            decision_net = layers_lib.max_pool2d(decision_net, [2, 2], scope='decision/pool6')

            decision_net = layers.conv2d(decision_net, 32, [5, 5], scope='decision/conv8')

            with tf.name_scope('decision/global_avg_pool'):
                avg_decision_net = keras.layers.GlobalAveragePooling2D()(decision_net)

            with tf.name_scope('decision/global_max_pool'):
                max_decision_net = keras.layers.GlobalMaxPooling2D()(decision_net)

            with tf.name_scope('decision/global_avg_pool'):
                avg_prob_net = keras.layers.GlobalAveragePooling2D()(net_prob_mat)

            with tf.name_scope('decision/global_max_pool'):
                max_prob_net = keras.layers.GlobalMaxPooling2D()(net_prob_mat)

            # adding avg_prob_net and max_prob_net may not be needed, but it doesen't hurt
            decision_net = tf.concat([avg_decision_net, max_decision_net, avg_prob_net, max_prob_net], axis=1)

            decision_net = layers.fully_connected(decision_net, 1, scope='decision/FC9',
                                                  normalizer_fn=None,
                                                  biases_initializer=tf.constant_initializer(0),
                                                  activation_fn=None)
        return decision_net


    def get_loss(self, net_model, masks, batch_size=None, return_segmentation_net=True, return_decision_net=True, output_resolution_reduction=8):
      """Adds all losses for the model.
    
      Note the final loss is not returned. Instead, the list of losses are collected
      by slim.losses. The losses are accumulated in tower_loss() and summed to
      calculate the total loss.
    
      Args:
        logits: List of logits from inference(). Each entry is a 2-D float Tensor.
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
        batch_size: integer
      """
      if not batch_size:
        raise Exception("Missing batch_size")

      net, decision_net, endpoints = net_model

      if output_resolution_reduction > 1:
        mask_blur_kernel = [output_resolution_reduction*2+1, output_resolution_reduction*2+1]
        masks = layers_lib.avg_pool2d(masks, mask_blur_kernel, stride=output_resolution_reduction, padding='SAME', scope='pool_mask',outputs_collections='tower_0/_end_points')

      if self.use_corss_entropy_seg_net is False:
          masks = tf.greater(masks, tf.constant(0.5))


      predictions = net

      tf.summary.image('prediction', predictions)

      l1 = None
      l2 = None

      if return_segmentation_net:
        if self.positive_weight > 1:
            pos_pixels = tf.less(tf.constant(0.0), masks)
            neg_pixels = tf.greater_equal(tf.constant(0.0), masks)

            num_pos_pixels = tf.cast(tf.count_nonzero(pos_pixels), dtype=tf.float32)
            num_neg_pixels = tf.cast(tf.count_nonzero(neg_pixels), dtype=tf.float32)

            pos_pixels = tf.cast(pos_pixels, dtype=tf.float32)
            neg_pixels = tf.cast(neg_pixels, dtype=tf.float32)

            positive_weight = tf.cond(num_pos_pixels > tf.constant(0,dtype=tf.float32),
                                      lambda: tf.multiply(tf.div(num_neg_pixels, num_pos_pixels),
                                                          tf.constant(self.positive_weight,dtype=tf.float32)),
                                      lambda: tf.constant(self.positive_weight, dtype=tf.float32))

            positive_weight = tf.reshape(positive_weight, [1])

            # weight positive samples more !!
            weights = tf.add(neg_pixels, tf.multiply(pos_pixels, positive_weight))

            # noramlize weights so that the sum of weights is always equal to the num of elements
            N = tf.constant(weights.shape[1]._value * weights.shape[2]._value, dtype=tf.float32)

            factor = tf.reduce_sum(weights,axis=[1,2])
            factor = tf.divide(N, factor)

            weights = tf.multiply(weights, tf.reshape(factor,[-1,1,1,1]))

            if self.use_corss_entropy_seg_net is False:
                l1 = tf.losses.mean_squared_error(masks, predictions, weights=weights)
            else:
                l1 = tf.losses.sigmoid_cross_entropy(logits=predictions, multi_class_labels=masks, weights=weights) # NOTE: weights were added but not tested yet !!
        else:
            if self.use_corss_entropy_seg_net is False:
                l1 = tf.losses.mean_squared_error(masks, predictions)
            else:
                l1 = tf.losses.sigmoid_cross_entropy(logits=predictions,multi_class_labels=masks)


      if return_decision_net:
          with tf.name_scope('decision'):
            masks = tf.cast(masks, tf.float32)
            label = tf.minimum(tf.reduce_sum(masks, [1, 2, 3]), tf.constant(1.0))

            if len(decision_net.shape) == 2:
                decision_net = tf.squeeze(decision_net, [1])
            elif len(decision_net.shape) == 4:
                decision_net = tf.squeeze(decision_net, [1, 2, 3])
            else:
                raise Exception("Only 2 or 4 dimensional output expected for decision_net")

            decision_net = tf.reshape(decision_net,[-1,1])
            label = tf.reshape(label, [-1, 1])

            l2 = tf.losses.sigmoid_cross_entropy(logits=decision_net,multi_class_labels=label, weights=self.decision_positive_weight)

      return [l1,l2]





    def _activation_summary(self, x):
      """Helper to create summaries for activations.
    
      Creates a summary that provides a histogram of activations.
      Creates a summary that measure the sparsity of activations.
    
      Args:
        x: Tensor
      """
      # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
      # session. This helps the clarity of presentation on tensorboard.
      tensor_name = re.sub('%s_[0-9]*/' % self.TOWER_NAME, '', x.op.name)
      tf.summary.histogram(tensor_name + '/activations', x)
      tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


    def _activation_summaries(self, endpoints):
      with tf.name_scope('summaries'):
        for act in endpoints.values():
          self._activation_summary(act)


    def restore(self, session, model_checkpoint_path, variables_to_restore = None, load_from_seg_only_net=False):

        if variables_to_restore is None:
            variables_to_restore = tf.trainable_variables()# + tf.moving_average_variables() # tf.moving_average_variables is required only in TF r1.1

        # this is only when loading from pre-trained network of segmetnation that did not have decision net layers
        # present at the same time
        if load_from_seg_only_net:
            variables_to_restore = [v for v in variables_to_restore if v.name.count('decision') == 0]

        saver = tf.train.Saver(variables_to_restore)
        try:
            saver.restore(session, model_checkpoint_path)

        except:
            # remove decision variables if cannot load them
            if type(variables_to_restore) is dict:
                variables_to_restore = [variables_to_restore[v] for v in variables_to_restore.keys() if v.find('decision') < 0]
            else:
                variables_to_restore = [v for v in variables_to_restore if v.name.find('decision') < 0]

            saver = tf.train.Saver(variables_to_restore)

            saver.restore(session, model_checkpoint_path)
