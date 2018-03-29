# -*- coding: utf-8 -*-
# @Time     : 2017/10/17  下午4:27
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : discriminator.py
# @Software : PyCharm

import tensorflow as tf
import CycleGAN.cycle_ops as ops

class Discriminator:
  def __init__(self, name, is_training, norm='instance', use_sigmoid=False):
    self.name = name
    self.is_training = is_training
    self.norm = norm
    self.reuse = False
    self.use_sigmoid = use_sigmoid

  def __call__(self, input):
    """
    Args:
      input: batch_size x image_size x image_size x 3
    Returns:
      output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
              filled with 0.9 if real, 0.0 if fake
    """
    with tf.variable_scope(self.name):
      # # convolution layers
      # C64 = ops.Ck(input, 64, reuse=self.reuse, norm=None,
      #     is_training=self.is_training, name='C64')             # (?, w/2, h/2, 64)
      # C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
      #     is_training=self.is_training, name='C128')            # (?, w/4, h/4, 128)
      # C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
      #     is_training=self.is_training, name='C256')            # (?, w/8, h/8, 256)
      # C512 = ops.Ck(C256, 512,reuse=self.reuse, norm=self.norm,
      #     is_training=self.is_training, name='C512')            # (?, w/16, h/16, 512)
      #
      # # apply a convolution to produce a 1 dimensional output (1 channel?)
      # # use_sigmoid = False if use_lsgan = True
      # output = ops.last_conv(C512, reuse=self.reuse,
      #     use_sigmoid=self.use_sigmoid, name='output')          # (?, w/16, h/16, 1)

      with tf.variable_scope('first', reuse=self.reuse):
          weights = tf.get_variable('weights',
                                    shape=[3, 1],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                                dtype=tf.float32))
          biases = tf.get_variable('biases',
                                   shape=[1],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))
          output = tf.matmul(input, weights) + biases

    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output
