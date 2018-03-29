# -*- coding: utf-8 -*-
# @Time     : 2017/10/17  下午7:58
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : emotion_tfr.py
# @Software : PyCharm

'''
This code is for our toy experiment in the paper
'''

import sys
import tensorflow as tf
import numpy as np
import xlrd
import matplotlib.pyplot as plt

excel_filename = '/Users/zhuxinyue/ML/tfrecords/data.xlsx'
tfrecords_filename_X = '/Users/zhuxinyue/ML/tfrecords/dataX.tfrecords'
tfrecords_filename_Y = '/Users/zhuxinyue/ML/tfrecords/dataY.tfrecords'
data_path_emotion = '/Users/zhuxinyue/ML/SFEW/train2/'

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def readexcel(filename, sheet):
    data = xlrd.open_workbook(filename)
    table = data.sheets()[sheet]
    print(table)
    nrows = table.nrows
    ncols = table.ncols
    print(nrows, ncols)
    array = np.zeros(shape=[nrows, ncols], dtype=np.float32)

    for i in range(nrows):
        for j in range(ncols):
            array[i, j] = table.row_values(i)[j]
    return array

def conver_to_tfrecord(tfrecords_filename, sheet):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    array = readexcel(excel_filename, sheet)
    print(array.shape)

    for i in range(array.shape[0]):
        if i % 10 == 0:
            print('tfrecording...', i)
        data = array[i]
        data_raw = data.tostring()
        # write to tfRecord
        example = tf.train.Example(features=tf.train.Features(feature={
            'data_raw': _bytes_feature(data_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def main(unused_argv):
  conver_to_tfrecord(tfrecords_filename_X, 0)
  conver_to_tfrecord(tfrecords_filename_Y, 1)
  print('Load successfully')

if __name__ == '__main__':
  tf.app.run()
