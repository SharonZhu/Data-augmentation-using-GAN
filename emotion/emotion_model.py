# -*- coding: utf-8 -*-
# @Time     : 2017/9/17  下午8:20
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : model.py.py
# @Software : PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ops import *

def inference(images, batch_size, n_classes):
    '''

    :param images:image batch, 4D tensor
    :param batch_size:
    :param n_classes: 50
    :return:
    output tensor with the computed logits, float, [batch_size,n_classes]
    '''

    # conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 1, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=5e-2, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pooling1 and norm1
    with tf.variable_scope('pooling1') as scope:
        pooling1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pooling1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 64, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=5e-2, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

        # pooling2 and norm2
        with tf.variable_scope('pooling2') as scope:
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm2')
            pooling2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                      padding='SAME', name='pooling2')

    # fc1
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pooling2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # fc2
    with tf.variable_scope('fc2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[256, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name='fc2')

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[256, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name='softmax_linear')

        # epsilon = tf.constant(0.001, shape=[n_classes])
        #
        # softmax_linear += epsilon
    return softmax_linear, fc2


def losses(logits,labels):
    '''
    compute losses from logits and labels
    :param logits:
    :param labels:
    :return: loss tensor of float type
    '''
    labels = tf.cast(labels,tf.float32)
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels,
                                                                name='xentropy_per_example')

        loss = tf.reduce_mean(cross_entropy,name='loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss

def training(loss,learning_rate):
    '''
    training ops
    :param loss:
    :param learning_rate:
    :return: train_op: the op for training
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy