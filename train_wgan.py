# -*- coding: utf-8 -*-
# @Time     : 2017/9/21  下午10:31
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : train_wgan.py
# @Software : PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from WGAN import WGAN
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

EPOCH = 2400 # 2399
BATCH_SIZE = 64
Z_DIM = 100
DATASET = 'Caltech101'
CHECKPOINT_DIR = 'wgan_faces/checkpoint/'
RESULT_DIR = '/Users/zhuxinyue/ML/Gen_Caltech101/faces/'
LOG_DIR = 'wgan_faces/log/'

"""main"""
def main():

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        gan = WGAN(classes='Faces', sess=sess,  epoch=EPOCH, batch_size=BATCH_SIZE,
                   z_dim=Z_DIM, dataset_name=DATASET, augmentation=False, aug_ratio=12,
                   checkpoint_dir=CHECKPOINT_DIR, result_dir=RESULT_DIR, log_dir=LOG_DIR)


        # build graph
        gan.build_model()

        # show network architecture
        # show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        # gan.visualize_results(EPOCH - 1)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()