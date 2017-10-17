# -*- coding: utf-8 -*-
# @Time     : 2017/10/17  下午2:47
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : cycleGAN.py
# @Software : PyCharm

from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *
from emotion.emotion_data import read_train_sets

class cycleGAN(object):
    def __init__(self, classes, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.classes = classes
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_name = "cycleWGAN"     # name for checkpoint

        if dataset_name == 'face_emotion':
            dataset_dir = '/Users/zhuxinyue/ML/' + dataset_name + '/'
            # parameters
            self.input_height = 48
            self.input_width = 48
            self.output_height = 48
            self.output_width = 48

            self.z_dim = z_dim  # dimension of noise-vector
            self.c_dim = 1  # dimension of channels?

            # WGAN parameter
            self.disc_iters = 5  # The number of critic iterations for one-step of generator

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load face_emotion
            self.data_X, self.data_y, _, _, _, _ = read_train_sets(dataset_dir, self.classes, 0)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size

        else:
            if dataset_name == 'generate':
                self.input_height = 48
                self.input_width = 48
                self.output_height = 48
                self.output_width = 48

                self.z_dim = z_dim  # dimension of noise-vector
                self.c_dim = 1  # dimension of channels?

                # WGAN parameter
                self.disc_iters = 5  # The number of critic iterations for one-step of generator

                # train
                self.learning_rate = 0.0002
                self.beta1 = 0.5

            else:
                raise NotImplementedError

    def discriminator(self, x, is_training=True, reuse=False):
        """
            Args:
              input: batch_size x image_size x image_size x 3
            Returns:
              output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                      filled with 0.9 if real, 0.0 if fake
            """
        with tf.variable_scope("discriminator", reuse=reuse):

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 2, 2, name='d_conv3'), is_training=is_training, scope='d_bn3'))
            net = lrelu(bn(conv2d(net, 512, 4, 4, 2, 2, name='d_conv4'), is_training=is_training, scope='d_bn4'))
            print(net)
            output = last_conv(net, reuse=reuse, name='output')  # (?, w/16, h/16, 1)
            print('discriminator out', output.shape)
            return output

    def generator(self, input, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.nn.relu(bn(conv2d(input, 32, 7, 7, 1, 1, name='g_conv1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(conv2d(net, 64, 3, 3, 2, 2, name='g_conv2'), is_training=is_training, scope='g_bn2'))
            net = tf.nn.relu(bn(conv2d(net, 128, 3, 3, 2, 2, name='g_conv3'), is_training=is_training, scope='g_bn3'))

            # resnet block
            res_output = n_res_blocks(net, reuse=reuse, n=6)  # (?, w/4, h/4, 128)

            # fractional-strided convolution
            u64 = uk(res_output, 64, reuse=reuse, name='u64')  # (?, w/2, h/2, 64)
            u32 = uk(u64, 32, reuse=reuse, name='u32', output_size=self.input_height)  # (?, w, h, 32)

            output = tf.nn.tanh(bn(conv2d(u32, 1, 7, 7, 1, 1, name='g_convout'), is_training=is_training, scope='g_bnout'))
            print('generator out', output.shape)

            return output

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images
        # D_real, D_real_logits, _ = self.discriminator(self.inputs, is_training=True, reuse=False)
        #
        # # output of D for fake images
        # G = self.generator(self.z, is_training=True, reuse=False)
        # D_fake, D_fake_logits, _ = self.discriminator(G, is_training=True, reuse=True)
        #
        # # get loss for discriminator
        # d_loss_real = - tf.reduce_mean(D_real)
        # d_loss_fake = tf.reduce_mean(D_fake)
        #
        # self.d_loss = d_loss_real + d_loss_fake
        #
        # # get loss for generator
        # self.g_loss = - d_loss_fake

        # generate
        fake = self.generator(self.z, is_training=True, reuse=False)

        # discriminate
        r_logit = self.discriminator(self.inputs, is_training=True, reuse=False)
        f_logit = self.discriminator(fake, reuse=True)

        # losses
        wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
        self.d_loss = -wd
        self.g_loss= -tf.reduce_mean(f_logit)

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            #           .minimize(self.d_loss, var_list=d_vars)
            # self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
            #           .minimize(self.g_loss, var_list=g_vars)

            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            # self.d_optim = optimizer.minimize(self.d_loss, var_list=d_vars)
            #
            # self.g_optim = optimizer.minimize(self.g_loss, var_list=g_vars)

            # compute gradient and vars
            self.d_grads_and_vars = optimizer.compute_gradients(self.d_loss, var_list=d_vars)
            self.g_grads_and_vars = optimizer.compute_gradients(self.g_loss, var_list=g_vars)

            # training ops
            self.d_optim = optimizer.apply_gradients(self.d_grads_and_vars)
            self.g_optim = optimizer.apply_gradients(self.g_grads_and_vars)


        # weight clipping
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        # d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        # d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        # d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        # g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)


        # final summary operations
        # self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        # self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

        self.d_sum = tf.summary.scalar("wd", wd)
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)

        for var in tf.trainable_variables():
            self.var_sum = tf.summary.histogram(var.op.name + '/values', var)

        for grad, var in self.d_grads_and_vars + self.g_grads_and_vars:
            self.grad_sum = tf.summary.histogram(var.op.name + "/gradients", grad)

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")


        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                # _, _, summary_str, d_loss = self.sess.run([self.d_optim, self.clip_D, self.d_sum, self.d_loss],
                #                                feed_dict={self.inputs: batch_images, self.z: batch_z})
                _, _, summary_str, d_loss = self.sess.run([self.d_optim, self.clip_D, self.summary_op, self.d_loss],
                                                          feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                if (counter-1) % self.disc_iters == 0 or counter==99:
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.summary_op, self.g_loss],
                                                           feed_dict={self.inputs: batch_images, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.jpg'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            # self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0