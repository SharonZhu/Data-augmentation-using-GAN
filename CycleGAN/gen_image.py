# -*- coding: utf-8 -*-
# @Time     : 2017/10/19  上午10:31
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : gen_image.py
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
import os
from CycleGAN.model import CycleGAN as CG
import CycleGAN.utils as utils
from scipy import misc
from emotion.emotion_data import read_train_sets

FLAGS = tf.flags.FLAGS

# tf.flags.DEFINE_string('model', 'pretrained/face2emotion.pb', 'model path (.pb)')
tf.flags.DEFINE_string('input', 'image_0001.jpg', 'input image path (.jpg)')
tf.flags.DEFINE_string('inputdir', '/Users/zhuxinyue/ML/Caltech101/Faces_easy/', 'input image dir')
# tf.flags.DEFINE_string('output', 'output_sample.jpg', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '48', 'image size, default: 256')

tf.flags.DEFINE_string('checkpoint_dir', 'checkpoints/20171018-1457/', 'checkpoints directory path')
tf.flags.DEFINE_string('result_dir', '/Users/zhuxinyue/ML/gen_CG/', 'checkpoints directory path')
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
              dataset_name='face_emotion',
              batch_size=1)

            input_image = tf.placeholder(tf.float32, shape=[1, FLAGS.image_size, FLAGS.image_size, 1], name='input_image')
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

            # _, face_images = load_caltech101(FLAGS.inputdir, FLAGS.image_size)
            classes = ['neutral']
            face_images, lables, _, _, _, _ = read_train_sets(FLAGS.inputdir, classes, 0, 5000)
            gen_num = face_images.shape[0]
            print(gen_num)

            for i in range(gen_num):
                print('Generating...', i)
                face_image = np.reshape(face_images[i], [1, FLAGS.image_size, FLAGS.image_size, 1])
                save_path = FLAGS.result_dir + 'genimg_' + str(i) + '.jpg'

                gen_image = cycle_gan.G.__call__(input_image)

                output = sess.run(gen_image, feed_dict={input_image: face_image})
                # print(output)
                output = np.reshape(output, [FLAGS.image_size, FLAGS.image_size])
                output = output * 255
                print(output.shape)
                # display_image(output, gray=True)
                # plt.show()

                misc.toimage(output).save(save_path)

            # display fake image
            # display_image(gen_image)
            # plt.show()
            # sys.exit()
        sess.close()

def main(unused_argv):
  restore_gen()

if __name__ == '__main__':
  tf.app.run()
