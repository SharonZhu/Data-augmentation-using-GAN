# -*- coding: utf-8 -*-
# @Time     : 2017/10/17  上午10:06
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : emotion_evaluate.py
# @Software : PyCharm

import sys

import tensorflow as tf
import numpy as np
from collections import Counter

from PIL import Image
import matplotlib.pyplot as plt
import emotion.emotion_model as model
import os
import emotion.emotion_data as data

data_path = '/Users/zhuxinyue/ML/face_emotion/'
logs_train_dir = '../emotion/logs/gan_real/'
logs_var_dir = '../emotion/logs/gan_real/'

BATCH_SIZE = 1
N_CLASSES = 6
image_size = 48
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

def evaluate_one_image(test_image, test_cls, logs_train_dir):
    # display image
    # data.display_image(test_image)

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_size, image_size, 1])

        test_image = np.reshape(test_image, [BATCH_SIZE, image_size, image_size, 1])

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


def main(argv=None):  # pylint: disable=unused-argument
    right_pred = 0
    right_pred_cls = np.zeros(shape=[N_CLASSES], dtype=np.int64)
    cls_accuracy = []

    test_images, test_clss = data.read_test_set(data_path, classes, 4000)
    test_num = test_images.shape[0]
    print('Number of test data:', test_num)

    # Calculate data numbers in each class
    cls_num = Counter(test_clss)
    print('cla_num:', cls_num)

    for num in range(test_num):
        pred_index, max_index = evaluate_one_image(test_images[num], test_clss[num], logs_train_dir)
        # print(test_clss[num])
        # print(pred_index)
        if pred_index == test_clss[num]:
            right_pred += 1
            right_pred_cls[max_index] += 1
            # print(right_pred_cls)

    accuracy = right_pred / test_num
    print('Final accuracy:', accuracy)

    for i in classes:
        index = classes.index(i)
        cls_accuracy.append(right_pred_cls[index] / cls_num[i])
        print(i, cls_accuracy[index], right_pred_cls[index], cls_num[i])
    print(cls_accuracy)
if __name__ == '__main__':
  tf.app.run()