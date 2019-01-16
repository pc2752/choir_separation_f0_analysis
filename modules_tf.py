from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import rnn
import config


tf.logging.set_verbosity(tf.logging.INFO)

def DeepConvSep(input_, feat_size = 513, time_context = 30):
    conv1 = tf.layers.conv2d(input_, 50, (1, feat_size), strides=(1, 1), padding='valid', name="F_1", activation = None)

    conv2 = tf.layers.conv2d(conv1, 50, (int(time_context/2),1), strides=(1,1),  padding = 'valid', name = "F_2", activation = None)

    encoded = tf.layers.dense(conv2, 128, name = "encoded")

    encoded_2 = tf.layers.dense(encoded, conv2_shape, name = "encoded_1")

    deconv1 = deconv2d(encoded_2, shape)

    deconv2 = deconv2d(deconv2, shape_2, activation=tf.nn.relu)

    return deconv2


def DeepSalience(input_, is_train):
    conv1 = tf.layers.conv2d(input_, 128, (5, 5), strides=(1, 1), padding='same', name="conv_1", activation = tf.nn.relu)

    conv1 = tf.layers.batch_normalization(conv1, training=is_train)

    conv2 = tf.layers.conv2d(conv1, 64, (5, 5), strides=(1, 1), padding='same', name="conv_2", activation = tf.nn.relu)

    conv2 = tf.layers.batch_normalization(conv2, training=is_train)

    conv3 = tf.layers.conv2d(conv2, 64, (3, 3), strides=(1, 1), padding='same', name="conv_3", activation = tf.nn.relu)

    conv3 = tf.layers.batch_normalization(conv3, training=is_train)

    conv4 = tf.layers.conv2d(conv3, 64, (3, 3), strides=(1, 1), padding='same', name="conv_4", activation = tf.nn.relu)

    conv4 = tf.layers.batch_normalization(conv4, training=is_train)

    conv5 = tf.layers.conv2d(conv4, 64, (3, 70), strides=(1, 1), padding='same', name="conv_5", activation = tf.nn.relu)

    conv5 = tf.layers.batch_normalization(conv5, training=is_train)

    final_layer = tf.layers.conv2d(conv5, 1, (1, 1), strides=(1, 1), padding='same', name="conv_6", activation = None)

    return tf.squeeze(final_layer)

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    # try:
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1], name = name)

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    # if with_w:
    #   return deconv, w, biases
    # else:
  return deconv
def selu(x):
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))




if __name__ == '__main__':
  main()