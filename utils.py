"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import matplotlib.pyplot as plt
from scipy import misc
import imgaug as ia
from imgaug import augmenters as iaa

import os
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import data

def display_image(image, gray=False):
    dis_image = image.astype(np.uint8)
    plt.figure()

    if gray:
        plt.imshow(dis_image, cmap='gray')
    else:
        plt.imshow(dis_image)

def load_caltech101(data_path, image_size):
    images = []
    labels = []
    files = os.listdir(data_path)

    for fl in files:
        file = data_path + fl
        image = misc.imread(file, mode='L')
        image = misc.imresize(image, [image_size, image_size])

        '''test image'''
        # print(image.shape)
        # display_image(image, True)
        # plt.show()
        # sys.exit()

        image = np.reshape(image, [image_size, image_size, -1])
        # print(image.shape)
        images.append(image)

        # label = 1.0
        # labels.append(label)

    images_uint = np.array(images, dtype=np.uint8)

    images = np.array(images, dtype=np.float32)
    # labels = np.array(labels, dtype=np.int64)

    print('Loding training data completed')
    print('images_shape:', images.shape)
    # print('labels_shape:', labels.shape)

    return images_uint, images/255

def data_augmentation(ratio, images):
    '''
    Augmentation of images [batch_size, h, w, channel]
    :param ratio: the number of augmentation of each image
    :param images:
    :return: [batch_size * ratio, h, w, channel], normed
    '''
    images_aug = None
    seq = []
    st = lambda aug: iaa.Sometimes(0.3, aug)
    for i in range(ratio):
        seq_one = iaa.Sequential([
            # iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            # iaa.Flipud(0.5),  # vertically flip 50% of all images
            # st(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
            # convert images into their superpixel representation
            st(iaa.Crop(percent=(0, 0.1))),  # crop images by 0-10% of their height/width
            # st(iaa.GaussianBlur((0, 3.0))),  # blur images with a sigma between 0 and 3.0
            st(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),  # sharpen images
            st(iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))),  # emboss images
            # search either for all edges or for directed edges
            st(iaa.Sometimes(0.5,
                             iaa.EdgeDetect(alpha=(0, 0.7)),
                             iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                             )),
            # st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)),  # add gaussian noise to images
            # st(iaa.Dropout((0.0, 0.1), per_channel=0.5)),  # randomly remove up to 10% of the pixels
            # st(iaa.Invert(0.25, per_channel=True)),  # invert color channels
            st(iaa.Add((-10, 10), per_channel=0.5)),  # change brightness of images (by -10 to 10 of original value)
            st(iaa.Multiply((0.5, 1.5), per_channel=0.5)),  # change brightness of images (50-150% of original value)
            st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),  # improve or worsen the contrast
            st(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_px={"x": (-16, 16), "y": (-16, 16)},  # translate by -16 to +16 pixels (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=ia.ALL,  # use any of scikit-image's interpolation methods
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # st(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25))
            # apply elastic transformations with random strengths
        ],
            random_order=True  # do all of the above in random order
        )
        seq.append(seq_one)
        if i == 0:
            images_aug = seq[0].augment_images(images)
        else:
            aug_temp = seq[i].augment_images(images)
            images_aug = np.concatenate((images_aug, aug_temp), axis=0)

    print('Augmentation shape', images_aug.shape)
    images_aug = np.array(images_aug, dtype=np.float32)
    return images_aug/255


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

""" Drawing Tools """
# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)