# -*- coding: utf-8 -*-
# @Time     : 2017/9/17  上午10:34
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : data.py.py
# @Software : PyCharm

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

def display_image(image):
    dis_image = image.astype(np.uint8)
    plt.figure()

    plt.imshow(dis_image)

def classes_extraction(path):
    classes = []
    for file in os.listdir(path):
        classes.append(file)
    classes = classes[1:]
    print('number of classes:', len(classes))
    print('classes:', classes)
    return classes

def load_data(data_set, data_path, classes, image_size):
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
            obj_num = int((fl.split('_')[-1]).split('.')[0])
            # throw away test data
            if (obj_num % 10) != 0 and data_set == 'train':
                image = misc.imread(fl)
                image = misc.imresize(image, [image_size, image_size])
                image = np.reshape(image, [image_size, image_size, -1])
                if image.shape[2] == 1:
                    # image = np.concatenate((image, np.empty(shape=[image_size, image_size, 2])), axis = 2)
                    continue
                images.append(image)

                label = np.zeros(len(classes), dtype=np.float32)
                label[index] = 1.0
                labels.append(label)

                cls.append(obj)
            else:
                if (obj_num % 10) == 0 and data_set == 'test':
                    image = misc.imread(fl)
                    image = misc.imresize(image, [image_size, image_size])
                    image = np.reshape(image, [image_size, image_size, -1])
                    if image.shape[2] == 1:
                        # image = np.concatenate((image, np.empty(shape=[image_size, image_size, 2])), axis=2)
                        continue
                    test_images.append(image)
                    test_cls.append(obj)

    if data_set == 'train':
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        cls = np.array(cls)

        print('Loding training data completed')
        print('images_shape:', images.shape)
        print('labels_shape:', labels.shape)

        return images, labels, cls

    if data_set == 'test':
        # Normalization
        test_images = np.array(test_images, dtype=np.uint8)
        test_images = test_images.astype(np.float32)
        test_images = test_images / 255
        print('Loading test data completed')

        return test_images, test_cls

# test training data
# data_path = '/Users/zhuxinyue/ML/Caltech101/'
# cls = classes_extraction(data_path)
# images, labels, cls = load_data('train', data_path, cls, 256)
# print(images.shape)
# print(cls[0])

def load_test(data_path, test_path, image_size):
    test_img = []
    test_cls = []

    print('Reading test images...')

    fr = open(test_path)
    for i in fr.readlines():
        item = i.split('/')
        object = item[0]
        content = item[1].strip('\n')

        img_path = data_path + object + '/' + content + '.jpeg'
        image = misc.imread(img_path)
        image = misc.imresize(image, [image_size, image_size])
        test_img.append(image)
        test_cls.append(object)

    # Normalization
    test_img = np.array(test_img, dtype=np.uint8)
    test_img_origin = test_img.astype(np.float32)
    test_img = test_img_origin / 255

    print('Loading test data completed')
    return test_img_origin, test_img, test_cls


# test loading data
# data_path = '/Users/zhuxinyue/ML/Corel5K/Corel5K/'
# train_path = '/Users/zhuxinyue/ML/Corel5K/Corel5K/corel5k_train_list.txt'
# test_path = '/Users/zhuxinyue/ML/Corel5K/Corel5K/corel5k_test_list.txt'
# classes = classes_extraction(data_path)
# image_size = 128

# images, labels, cls = load_train(data_path, train_path, classes, image_size)
# print('labels:', labels.shape)
# print('images:', images.shape)
# print(labels[0])
# print(cls[0])
# display_image(images[0])
# plt.show()

# images, clss = load_test(data_path, test_path, image_size)
# print(images[0])
# display_image(images[50])
# plt.show()

def read_train_sets(data_path, image_size, classes, validation_size):

  images, labels, cls = load_data('train', data_path, classes, image_size)
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


def read_test_set(data_path, classes, image_size):
  images, cls  = load_data('test', data_path, classes, image_size)
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