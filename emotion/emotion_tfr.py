# -*- coding: utf-8 -*-
# @Time     : 2017/10/17  下午7:58
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : emotion_tfr.py
# @Software : PyCharm

import sys
import tensorflow as tf
import numpy as np
import emotion.emotion_data as data
from utils import load_caltech101
import matplotlib.pyplot as plt

tfrecords_filename_disgust = '/Users/zhuxinyue/ML/SFEW/tfrecords/disgust.tfrecords'
tfrecords_filename_sad = '/Users/zhuxinyue/ML/SFEW/tfrecords/sad.tfrecords'
tfrecords_filename_angry = '/Users/zhuxinyue/ML/SFEW/tfrecords/angry.tfrecords'
tfrecords_filename_neutral = '/Users/zhuxinyue/ML/SFEW/tfrecords/neutral.tfrecords'
tfrecords_filename_fear = '/Users/zhuxinyue/ML/SFEW/tfrecords/fear.tfrecords'
tfrecords_filename_happy = '/Users/zhuxinyue/ML/SFEW/tfrecords/happy.tfrecords'
tfrecords_filename_surprise = '/Users/zhuxinyue/ML/SFEW/tfrecords/surprise.tfrecords'
data_path_emotion = '/Users/zhuxinyue/ML/SFEW/train2/'

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def conver_to_tfrecord(set, tfrecords_filename, data_path, classes):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    if set == 'train':
        mode = tfrecords_filename.split('/')[-1].split('.')[0]
        print(mode)
        images, lables, _, _, _, _ = data.read_train_sets(data_path, classes, 0, 3)
        # if mode == 'disgust':
        #     images, lables, _, _, _, _ = data.read_train_sets(data_path, classes, 0, None)
        #     # print(images.shape)
        # else:
        #     if mode == 'neutral':
        #         images, lables, _, _, _, _ = data.read_train_sets(data_path, classes, 0, None)
        #         # print(images.shape)
        #     else:
        #         if mode == 'sad' or mode == 'angry':
        #             images, lables, _, _, _, _ = data.read_train_sets(data_path, classes, 0, None)
        #         else:
        #             if mode == 'fear':
        #                 images, lables, _, _, _, _ = data.read_train_sets(data_path, classes, 0, None)
        #             else:
        #                 if mode == 'happy':
        #                     images, lables, _, _, _, _ = data.read_train_sets(data_path, classes, 0, None)
        #                 else:
        #                     if mode == 'surprise':
        #                         images, lables, _, _, _, _ = data.read_train_sets(data_path, classes, 0, None)
        #                            else:
        #                                 images, labels = [None, None]
    else:
        if set == 'test':
            images, labels = data.read_test_set(data_path, classes, 3000)
        else:
            images, labels = [None, None]

    set_num = images.shape[0]

    for i in range(set_num):
        if i % 10 == 0:
            print('tfrecording...', i)
        img_raw = images[i].tostring()
        # label_raw = labels[i]

        # write to tfRecord
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': _bytes_feature(img_raw),
            # 'ann_raw': _int64_feature(label_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def main(unused_argv):
  print("Convert emotion data to tfrecords...")
  # classes_emotion = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
  classes_emotion = ['neutral']
  conver_to_tfrecord('train', tfrecords_filename_neutral, data_path_emotion, classes_emotion)
  print('Load successfully')

  classes_emotion = ['angry']
  conver_to_tfrecord('train', tfrecords_filename_angry, data_path_emotion, classes_emotion)
  print('Load successfully')

  classes_emotion = ['disgust']
  conver_to_tfrecord('train', tfrecords_filename_disgust, data_path_emotion, classes_emotion)
  print('Load successfully')

  classes_emotion = ['fear']
  conver_to_tfrecord('train', tfrecords_filename_fear, data_path_emotion, classes_emotion)
  print('Load successfully')

  classes_emotion = ['happy']
  conver_to_tfrecord('train', tfrecords_filename_happy, data_path_emotion, classes_emotion)
  print('Load successfully')

  classes_emotion = ['sad']
  conver_to_tfrecord('train', tfrecords_filename_sad, data_path_emotion, classes_emotion)
  print('Load successfully')

  classes_emotion = ['surprise']
  conver_to_tfrecord('train', tfrecords_filename_surprise, data_path_emotion, classes_emotion)
  print('Load successfully')

  # print("Convert face data to tfrecords...")
  # # classes_faces = ['Faces_easy']
  # # conver_to_tfrecord('train', tfrecords_filename_faces, data_path_faces, classes_faces)
  # classes_emotion = ['neutral']
  # conver_to_tfrecord('train', tfrecords_filename_neutral, data_path_emotion, classes_emotion)

if __name__ == '__main__':
  tf.app.run()
