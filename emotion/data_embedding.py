# -*- coding: utf-8 -*-
# @Time     : 2017/10/14  下午9:23
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : data_embedding.py
# @Software : PyCharm

import sys
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.tensorboard.plugins import projector

from emotion.emotion_data import read_train_sets

data_path = '/Users/zhuxinyue/ML/face_emotion/'
log_dir = 'embeddings'
# classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
classes = ['angry', 'happy']

images, labels, _, _, _, _ = read_train_sets(data_path, classes, 0, 6000)
images = np.reshape(images, [-1, 48*48*1])
images = tf.Variable(images, name='images')

# move this tsv file to embeddings after training
metadata = os.path.join('metadata.tsv')
labels = np.argmax(labels, axis=1)

with open(metadata, 'w') as metadata_file:
    for row in labels:
        metadata_file.write('%d\n' % row)
    print('Done metadata')

with tf.Session() as sess:
    saver = tf.train.Saver([images])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(log_dir, 'images.ckpt'))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    embedding.metadata_path = metadata
    projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)
