# -*- coding: utf-8 -*-
# @Time     : 2017/10/16  下午4:58
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : emotion_data.py
# @Software : PyCharm

'''
Emotion dataset comes from https://github.com/sjchoi86/img_dataset
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import numpy as np
# import cv2
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.utils import shuffle
import tensorflow as tf
from collections import Counter

def display_image(image):
    dis_image = image.astype(np.uint8)
    plt.figure()

    plt.imshow(dis_image)


def load_caltech101(data_path, image_size):
    images = []
    labels = []
    files = os.listdir(data_path)

    for fl in files:
        file = data_path + fl
        image = misc.imread(file, mode='L')
        image = misc.imresize(image, [image_size, image_size])

        '''test image'''
        # print(image.shape)
        # display_image(image, True)
        # plt.show()
        # sys.exit()

        image = np.reshape(image, [image_size, image_size, -1])
        # print(image.shape)
        images.append(image)

        # label = 1.0
        # labels.append(label)

    images_uint = np.array(images, dtype=np.uint8)

    images = np.array(images, dtype=np.float32)
    # labels = np.array(labels, dtype=np.int64)

    print('Loding training data completed')
    print('images_shape:', images.shape)
    # print('labels_shape:', labels.shape)

    return images_uint, images/255

def load_data(data_set, data_path, classes, bound1, bound2):
    images = []
    test_images = []
    test_cls = []
    labels = []
    cls = []

    print('Reading training images...')

    for obj in classes:
        index = classes.index(obj)
        print('Loading {} files (index:{})'.format(obj, index))

        path = os.path.join(data_path, obj, '*g')
        files = glob.glob(path)

        for fl in files:
            # print(fl)
            obj_num = int((fl.split('/')[-1]).split('.')[0])
            # throw away test data
            if obj_num < bound1 and data_set == 'train':
                image = misc.imread(fl)
                image = np.reshape(image, [48, 48, 1])
                images.append(image)

                label = np.zeros(len(classes), dtype=np.float32)
                label[index] = 1.0
                labels.append(label)

                cls.append(obj)
            else:
                if obj_num < bound2 and data_set == 'test':
                    image = misc.imread(fl)
                    image = np.reshape(image, [48, 48, 1])
                    test_images.append(image)
                    test_cls.append(obj)

    if data_set == 'train':
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        cls = np.array(cls)

        print('Loding training data completed')
        print('images_shape:', images.shape)
        print('labels_shape:', labels.shape)

        # Calculate data numbers in each class
        cls_num = Counter(cls)
        print(cls_num)
        return images, labels, cls

    if data_set == 'test':
        # Normalization
        test_images = np.array(test_images, dtype=np.uint8)
        test_images = test_images.astype(np.float32)
        test_images = test_images / 255
        print('Loading test data completed')

        return test_images, test_cls

def read_train_sets(data_path, classes, validation_size, bound_train):

  images, labels, cls = load_data('train', data_path, classes, bound_train, 0)
  images, labels, cls = shuffle(images, labels, cls)  # shuffle the data

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  if validation_size != 0:
      val_images = images[:validation_size]
      val_images = val_images / 255
      val_labels = labels[:validation_size]
      val_cls = cls[:validation_size]
  else:
      val_images, val_labels, val_cls = [None, None, None]
  train_images = images[validation_size:]
  train_images = train_images / 255
  train_labels = labels[validation_size:]
  train_cls = cls[validation_size:]

  return train_images, train_labels, train_cls, val_images, val_labels, val_cls


def read_test_set(data_path, classes, bound_test):
  images, cls  = load_data('test', data_path, classes, 0, bound_test)
  return images, cls

def inputs(image, label, cls, batch_size, capacity, min_after_dequeue):
    '''
    Input a batch of data
    '''
    with tf.name_scope('input'):
        input_queue = tf.train.slice_input_producer([image, label, cls])
        image = input_queue[0]
        label = input_queue[1]
        cls = input_queue[2]
        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, labels, clss= tf.train.shuffle_batch(
            [image, label, cls], batch_size=batch_size, num_threads=2,
            capacity=capacity,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=min_after_dequeue)

        return images, labels, clss