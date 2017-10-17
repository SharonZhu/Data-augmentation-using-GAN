# -*- coding: utf-8 -*-
# @Time     : 2017/9/19  上午9:01
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : evaluate_one_image.py
# @Software : PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc

import data
import model

data_path = '/Users/zhuxinyue/ML/airplanes_faces/'
logs_train_dir = 'logs/train'
logs_val_dir = 'logs/val'

BATCH_SIZE = 1
N_CLASSES = 2
image_size = 64
# classes = data.classes_extraction(data_path)
classes = ['airplanes', 'Faces']

def get_one_image(data_path, classes, image_size):
    test_images, test_clss = data.read_test_set(data_path, classes, image_size)
    num = test_images.shape[0]
    test_image_num = np.random.randint(0, num)
    test_image = test_images[test_image_num]
    test_cls = test_clss[test_image_num]
    print('Get Image! The Class is: ', test_cls)

    return test_image, test_cls

def evaluate_one_image(test_image, test_cls, logs_train_dir):
    # display image
    # data.display_image(test_image)

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_size, image_size, 3])

        test_image = np.reshape(test_image, [BATCH_SIZE, image_size, image_size, 3])

        logit = model.inference(test_image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("successful loading,global step is %s" % global_step)
            else:
                print("no checkpoint file founded")
                return
            prediction = sess.run(logit, feed_dict={x: test_image})
            # print(prediction)
            max_index = np.argmax(prediction)
            pred_index = classes[max_index]

            print('The true label is:', test_cls)
            print('The predict label is:', pred_index)
            # plt.show()
    return pred_index, max_index

def evaluate_one_image_with_dir():
    img_dir = input("Enter the dir of your image: ")
    image = Image.open(img_dir)
    image.show(image)
    image = image.resize([68, 68])
    image_array= np.array(image)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 1000

        image = tf.cast(image_array, tf.float32)
        image = tf.reshape(image,[68,68,1])
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 68, 68, 1])

        logit = model.inference(image,BATCH_SIZE,N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32,shape=[68,68])

        logs_train_dir = "D:/APP/PyCharm 2017.1/task/logs_Corel50K/"
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                print("successful loading,global step is %s"%global_step)
            else:
                print("no checkpoint file founded")
                return
            prediction = sess.run(logit,feed_dict={x:image_array})
            #print(prediction)
            max_index = np.argmax(prediction)
            pred_index = classes[max_index]
            print(max_index)

def main(argv=None):  # pylint: disable=unused-argument
    test_image, test_cls = get_one_image(data_path, classes, image_size)
    evaluate_one_image(test_image, test_cls, logs_train_dir)


if __name__ == '__main__':
  tf.app.run()