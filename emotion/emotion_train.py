# -*- coding: utf-8 -*-
# @Time     : 2017/9/17  下午8:31
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : train.py
# @Software : PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import tensorflow as tf
import numpy as np
from emotion import emotion_data as data
import emotion.emotion_model as model

data_path = '/Users/zhuxinyue/ML/SFEW/train2/'
logs_train_dir = 'logs2/gan/'
logs_train_dir2 = 'logs2/gan/'
logs_val_dir2 = 'logs2/gan/'
gen_path = ['/Users/zhuxinyue/ML/SFEW/generate/angry/',
            '/Users/zhuxinyue/ML/SFEW/generate/disgust/',
            '/Users/zhuxinyue/ML/SFEW/generate/fear/',
            '/Users/zhuxinyue/ML/SFEW/generate/happy/',
            '/Users/zhuxinyue/ML/SFEW/generate/sad/',
            '/Users/zhuxinyue/ML/SFEW/generate/surprise/']

IMG_H = 48
IMG_W = 48

classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
N_CLASSES = 6

BATCH_SIZE = 32
CAPACITY = 1000 + 3*BATCH_SIZE
MIN_AFTER_DEQUEUE = 500
MAX_STEP = 10000
learning_rate = 0.001

def run_training():
    # load data
    # tra_img, tra_label, tra_cls, val_img, val_label, val_cls = data.read_train_sets(
    #     data_path, classes, validation_size=0, bound_train=None)

    # angry_img, angry_label, angry_cls = data.load_gan_image(gen_path[0], classes, 'angry')
    disgust_img, disgust_label, disgust_cls = data.load_gan_image(gen_path[1], classes, 'disgust', channel=1)
    fear_img, fear_label, fear_cls = data.load_gan_image(gen_path[2], classes, 'fear', channel=1)
    # happy_img, happy_label, happy_cls = data.load_gan_image(gen_path[3], classes, 'happy')
    # sad_img, sad_label, sad_cls = data.load_gan_image(gen_path[4], classes, 'sad')
    # surprise_img, surprise_label, surprise_cls = data.load_gan_image(gen_path[5], classes, 'surprise')

    tra_img, tra_label, tra_cls, val_img, val_label, val_cls = data.read_train_sets(
        data_path, classes, validation_size=0.05, channel=1)

    concat_images = np.concatenate([tra_img, disgust_img, fear_img],
                                   axis=0)
    concat_labels = np.concatenate([tra_label, disgust_label, fear_label], axis=0)
    concat_clss = np.concatenate([tra_cls, disgust_cls, fear_cls],
                                 axis=0)

    # print(gan_label[0])
    # concat_images = np.concatenate([tra_img, gan_img])
    # concat_lables = np.concatenate([tra_label, gan_label])
    # concat_clss = np.concatenate([tra_cls, gen_cls])
    # print(concat_images.shape)
    # print(concat_labels.shape)
    # print(concat_labels[1000])

    # train_images, train_labels, train_clss= data.inputs(tra_img, tra_label, tra_cls,
    #                                          BATCH_SIZE, CAPACITY, MIN_AFTER_DEQUEUE)
    train_images, train_labels, train_clss = data.inputs(concat_images, concat_labels, concat_clss,
                                                         BATCH_SIZE, CAPACITY, MIN_AFTER_DEQUEUE)
    val_images, val_labels, val_clss= data.inputs(val_img, val_label, val_cls,
                                             BATCH_SIZE, CAPACITY, MIN_AFTER_DEQUEUE)
    print(train_images.shape)
    print(train_labels.shape)

    train_logits, _ = model.inference(train_images, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_labels)
    train_op = model.training(train_loss, learning_rate)
    train_eval = model.evaluation(train_logits, train_labels)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir2,sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir2,sess.graph)

        x_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_H, IMG_W, 1])
        y_true = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES])
        # y_true_cls = tf.argmax(y_true, dimension=1)

        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("successful loading,global step is %s" % global_step)
        else:
            print("no checkpoint file founded")

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                # test values of label and logits in a batch:
                tra_images, tra_labels, tra_clss = sess.run([train_images, train_labels, train_clss])
                _, tra_loss, tra_eval = sess.run([train_op, train_loss, train_eval],
                                                 feed_dict={x_image: tra_images,
                                                            y_true: tra_labels})
                # print('logits', tra_logits)
                # print('ce', tra_ce)
                # print(tra_loss)
                # sys.exit()
                if step % 10 == 0:
                    print('Step %d, train loss=%.4f, train accuracy=%.2f%%' % (step, tra_loss, tra_eval * 100))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                if step % 100 == 0 or (step + 1) == MAX_STEP:
                    vali_images, vali_labels, vali_clss = sess.run([val_images, val_labels, val_clss])
                    val_loss, val_acc = sess.run([train_loss, train_eval],
                                                 feed_dict={x_image: vali_images,
                                                            y_true: vali_labels})
                    print('**  Step %d, val loss = %.4f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, step)
                if step % 500 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir2, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training')
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


def main(argv=None):  # pylint: disable=unused-argument
    run_training()


if __name__ == '__main__':
    tf.app.run()