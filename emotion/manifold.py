# -*- coding: utf-8 -*-
# @Time     : 2017/10/30  下午3:52
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : manifold.py
# @Software : PyCharm

from time import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import(manifold, datasets, decomposition, ensemble,
                    discriminant_analysis, random_projection)
from emotion.emotion_data import read_train_sets
import tensorflow as tf
import emotion.emotion_model as model
import emotion.emotion_data as data

data_path = '/Users/zhuxinyue/ML/face_emotion/'
gen_path1 = '/Users/zhuxinyue/ML/gen_CG/disgust/'
gen_path2 = '/Users/zhuxinyue/ML/gen_CG/sad/'
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# classes = ['angry', 'happy', ' fear', 'surprise']

#
images, labels, _, _, _, _ = read_train_sets(data_path, classes, 0, 6000)
disgust_img, disgust_label, _ = data.load_gan_image(gen_path1, classes, 'disgust')
# sad_img, sad_label, _ = data.load_gan_image(gen_path2, classes, 'sad')

concat_images = np.concatenate([images, disgust_img],
                                   axis=0)
concat_labels = np.concatenate([labels, disgust_label], axis=0)


print(concat_images.shape)
# images = np.reshape(images, [-1, 48, 48])
X = np.reshape(concat_images, [-1, 48*48*1])
y = np.argmax(concat_labels, axis=1)
n_samples, n_featues = X.shape
n_neighbors = 30

# digits = datasets.load_digits(n_class=6)
# X = digits.data
# y = digits.target
# images = digits.images
# print(X.shape)
# print(y.shape)

print(images[0].shape)



#----------------------------------------------------------------------
# read the fc2 layer as input map
logs_train_dir = '../emotion/logs/gan_train/'
with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[n_samples, 48, 48, 1])
    _, map = model.inference(x, n_samples, 7)
    print(map.shape)
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

        maps = sess.run(map, feed_dict={x:concat_images})
        print(maps.shape)
        # plt.show()
#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    # ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(X.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


#----------------------------------------------------------------------
# Plot images of the digits
# n_img_per_row = 20
# img = np.zeros((100 * n_img_per_row, 100 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         # print(iy)
#         # print(img[ix:ix + 48, iy:iy + 48].shape)
#         img[ix:ix + 48, iy:iy + 48] = X[i * n_img_per_row + j].reshape((48, 48))
# print('flag')
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.title('A selection from the 64-dimensional digits dataset')


#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(maps)
plot_embedding(X_projected, "Random Projection of the digits")


#----------------------------------------------------------------------
# Projection on to the first 2 principal components

print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(maps)
plot_embedding(X_pca,
               "Principal Components projection of the digits (time %.2fs)" %
               (time() - t0))


#----------------------------------------------------------------------
# t-SNE embedding of the digits dataset

print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(maps)

plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

plt.show()