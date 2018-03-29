# -*- coding: utf-8 -*-
# @Time     : 2017/10/17  下午2:47
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : CycleGAN.py
# @Software : PyCharm

'''
CycleGAN codes partly come from https://github.com/vanhuyz/CycleGAN-TensorFlow
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import tensorflow as tf
import numpy as np

sys.path.append('../CycleGAN/')
import cycle_ops as ops
sys.path.append('../')
from utils import *
# from emotion.emotion_data import read_train_sets
from discriminator import Discriminator
from generator import Generator
from reader import Reader



REAL_LABEL = 0.9

class CycleGAN(object):
    def __init__(self, X_train_file, Y_train_file, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.model_name = "cycleWGAN"     # name for checkpoint
        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        self.norm = 'instance'
        self.X_train_file = ''
        self.Y_train_file = ''

        if dataset_name == 'face_emotion':
            self.X_train_file = X_train_file
            self.Y_train_file = Y_train_file
            dataset_dir = '/Users/zhuxinyue/ML/' + dataset_name + '/'
            # parameters
            self.image_size = 48

            self.c_dim = 1  # dimension of channels?
            self.ngf = 64

            # CycleGAN parameter
            self.disc_iters = 5  # The number of critic iterations for one-step of generator

            # train
            self.learning_rate = 0.0002
            self.lambda1 = 10.0,
            self.lambda2 = 10.0,
            self.beta1 = 0.5

        else:
            if dataset_name == 'sfew':
                self.X_train_file = X_train_file
                self.Y_train_file = Y_train_file
                # dataset_dir = '/Users/zhuxinyue/ML/' + dataset_name + '/'

                # parameters
                self.image_size = 128

                self.c_dim = 3  # dimension of channels?
                self.ngf = 64

                # CycleGAN parameter
                self.disc_iters = 5  # The number of critic iterations for one-step of generator

                # train
                self.learning_rate = 0.0001
                self.lambda1 = 10.0,
                self.lambda2 = 10.0,
                self.beta1 = 0.5

            else:
                if dataset_name == 'test':
                    self.X_train_file = X_train_file
                    self.Y_train_file = Y_train_file
                    # dataset_dir = '/Users/zhuxinyue/ML/' + dataset_name + '/'

                    # parameters
                    self.image_size = None

                    self.c_dim = 3  # dimension of channels?
                    self.ngf = 64

                    # CycleGAN parameter
                    self.disc_iters = 5  # The number of critic iterations for one-step of generator

                    # train
                    self.learning_rate = 0.0001
                    self.lambda1 = 10.0,
                    self.lambda2 = 10.0,
                    self.beta1 = 0.5
                else:
                    raise NotImplementedError

        # Generator X --> Y
        self.G = Generator('G', self.is_training, ngf=self.ngf, norm=self.norm, image_size=self.image_size)
        # Discriminator Y
        self.D_Y = Discriminator('D_Y',
                                 self.is_training, norm=self.norm, use_sigmoid=False)
        # Generator Y --> X
        self.F = Generator('F', self.is_training, norm=self.norm, image_size=self.image_size)
        # Discriminator X
        self.D_X = Discriminator('D_X',
                                 self.is_training, norm=self.norm, use_sigmoid=False)

        # self.fake_x = tf.placeholder(tf.float32,
        #                              shape=[batch_size, self.image_size, self.image_size, 3])
        # self.fake_y = tf.placeholder(tf.float32,
        #                              shape=[batch_size, self.image_size, self.image_size, 3])
        self.fake_x = tf.placeholder(tf.float32,
                                     shape=[batch_size, 3])
        self.fake_y = tf.placeholder(tf.float32,
                                     shape=[batch_size, 3])


    def model(self):
        if self.X_train_file != None and self.Y_train_file != None:
            X_reader = Reader(self.X_train_file, name='X',
                              image_size=self.image_size, batch_size=self.batch_size)
            Y_reader = Reader(self.Y_train_file, name='Y',
                              image_size=self.image_size, batch_size=self.batch_size)

            x = X_reader.feed()
            y = Y_reader.feed()

        else:
            x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 3])
            y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 3])
        """ Loss Function """

        cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)
        print(cycle_loss)

        # X -> Y
        fake_y = self.G(x)  # __call__(input)
        print(fake_y)
        G_gan_loss = self.generator_loss(self.D_Y, fake_y, use_lsgan=True)
        print(G_gan_loss)
        G_loss = G_gan_loss + cycle_loss
        D_Y_loss = self.discriminator_loss(self.D_Y, y, self.fake_y, use_lsgan=True)

        # Y -> X
        fake_x = self.F(y)
        print(fake_x)
        F_gan_loss = self.generator_loss(self.D_X, fake_x, use_lsgan=True)
        F_loss = F_gan_loss + cycle_loss
        D_X_loss = self.discriminator_loss(self.D_X, x, self.fake_x, use_lsgan=True)

        # summary
        tf.summary.histogram('D_Y/true', self.D_Y(y))
        tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))
        tf.summary.histogram('D_X/true', self.D_X(x))
        tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))

        tf.summary.scalar('loss/G', G_gan_loss)
        tf.summary.scalar('loss/D_Y', D_Y_loss)
        tf.summary.scalar('loss/F', F_gan_loss)
        tf.summary.scalar('loss/D_X', D_X_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)

        return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x

    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                              decay_steps, end_learning_rate,
                                              power=1.0),
                    starter_learning_rate
                )

            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                    .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step

        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
        F_optimizer = make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

        with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
            return tf.no_op(name='optimizers')

    '''Loss Function'''
    def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
        """ Note: default: D(y).shape == (batch_size,5,5,1),
                           fake_buffer_size=50, batch_size=1
        Args:
          G: generator object
          D: discriminator object
          y: 4D tensor (batch_size, image_size, image_size, 3)
        Returns:
          loss: scalar
        """
        if use_lsgan:
            # use mean squared error
            error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
            error_fake = tf.reduce_mean(tf.square(D(fake_y)))
        else:
            # use cross entropy
            error_real = -tf.reduce_mean(ops.safe_log(D(y)))
            error_fake = -tf.reduce_mean(ops.safe_log(1 - D(fake_y)))
        loss = (error_real + error_fake) / 2
        return loss

    def generator_loss(self, D, fake_y, use_lsgan=True):
        """  fool discriminator into believing that G(x) is real
        """
        if use_lsgan:
            # use mean squared error
            loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
        else:
            # heuristic, non-saturating loss
            loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
        return loss

    def cycle_consistency_loss(self, G, F, x, y):
        """ cycle consistency loss (L1 norm)
        """
        forward_loss = tf.reduce_mean(tf.abs(F(G(x)) - x))
        backward_loss = tf.reduce_mean(tf.abs(G(F(y)) - y))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        loss = tf.reshape(loss, [])
        return loss