# -*- coding: utf-8 -*-
# @Time     : 2017/11/11  上午9:13
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : gen_data.py
# @Software : PyCharm

"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/face2emotion.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 48
"""
import sys
import tensorflow as tf
import numpy as np
import xlsxwriter as xw
import os
from CycleGAN.model import CycleGAN as CG
from CycleGAN.data_tfr import readexcel

FLAGS = tf.flags.FLAGS

# tf.flags.DEFINE_string('model', 'pretrained/face2emotion.pb', 'model path (.pb)')
tf.flags.DEFINE_string('inputdir', '/Users/zhuxinyue/ML/tfrecords/data.xlsx', 'input image dir')
# tf.flags.DEFINE_string('output', 'output_sample.jpg', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '128', 'image size, default: 256')
tf.flags.DEFINE_string('checkpoint_dir', 'checkpoints/20171110-2239/', 'checkpoints directory path')
tf.flags.DEFINE_string('result_dir', '/Users/zhuxinyue/ML/tfrecords/data_gen.xlsx', 'checkpoints directory path')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')

import matplotlib.pyplot as plt
def display_image(image, gray=False):
    dis_image = image.astype(np.uint8)
    plt.figure()

    if gray:
        plt.imshow(dis_image, cmap='gray')
    else:
        plt.imshow(dis_image)


def restore_gen():
    '''test single image'''
    # image = misc.imread(FLAGS.input, mode='L')
    # print(image.shape)
    # image = misc.imresize(image, [FLAGS.image_size, FLAGS.image_size])
    # print(image.shape)
    # image = np.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, -1])
    # print(image.shape)
    # image = image / 255

    with tf.Graph().as_default():
        with tf.Session() as sess:

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)

            cycle_gan = CG(
              X_train_file='',
              Y_train_file='',
              dataset_name='test',
              batch_size=1)

            input_data = tf.placeholder(tf.float32, shape=[1, 3], name='input_image')
            cycle_gan.model()

            saver = tf.train.Saver()
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("successful loading,global step is %s" % global_step)
            else:
                print("no checkpoint file founded")
                return

            X = readexcel(FLAGS.inputdir, 0)
            gen_num = X.shape[0]

            wb = xw.Workbook(FLAGS.result_dir)
            sheet1 = wb.add_worksheet()

            for i in range(gen_num):
                print('Generating...', i)
                x = X[i]
                x = np.reshape(x, [1,3])
                gen_y = cycle_gan.G.__call__(x)

                output = sess.run(gen_y, feed_dict={input_data: x})
                print(output)
                for j in range(3):
                    sheet1.write(i, j, str(output[0, j]))

            wb.close()
        sess.close()

def main(unused_argv):
  restore_gen()

if __name__ == '__main__':
  tf.app.run()
