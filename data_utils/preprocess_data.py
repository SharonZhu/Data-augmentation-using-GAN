# -*- coding: utf-8 -*-
# @Time     : 2017/9/25  上午10:24
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : preprocess_data.py
# @Software : PyCharm

import imgaug as ia
from imgaug import augmenters as iaa
from utils import load_caltech101
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

data_path = '/Users/zhuxinyue/ML/Caltech101/brain/'

images, _, labels = load_caltech101(data_path, 64)
print(images.dtype)
images_aug = None
seq = []
st = lambda aug: iaa.Sometimes(0.3, aug)
for i in range(5):
    seq_one = iaa.Sequential([
            # iaa.Fliplr(0.5), # horizontally flip 50% of all images
            # iaa.Flipud(0.5), # vertically flip 50% of all images
            # st(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
            st(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
            # # st(iaa.GaussianBlur((0, 3.0))), # blur images with a sigma between 0 and 3.0
            st(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))), # sharpen images
            st(iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))), # emboss images
            # # search either for all edges or for directed edges
            st(iaa.Sometimes(0.5,
                iaa.EdgeDetect(alpha=(0, 0.7)),
                iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
            )),
            # # st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)), # add gaussian noise to images
            # # st(iaa.Dropout((0.0, 0.1), per_channel=0.5)), # randomly remove up to 10% of the pixels
            # st(iaa.Invert(0.25, per_channel=True)), # invert color channels
            st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
            st(iaa.Multiply((0.5, 1.5), per_channel=0.5)), # change brightness of images (50-150% of original value)
            st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
            st(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=ia.ALL, # use any of scikit-image's interpolation methods
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # # st(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)) # apply elastic transformations with random strengths
        ],
        random_order=True # do all of the above in random order
    )
    seq.append(seq_one)
    if i == 0:
        images_aug = seq[0].augment_images(images)
    else:
        aug_temp = seq[i].augment_images(images)
        images_aug = np.concatenate((images_aug, aug_temp), axis=0)
    print(images_aug.shape)

print(images_aug.shape)
print(images_aug.dtype)

print('0', images_aug[0].shape)
print('1', images_aug[1].shape)
plt.figure(1)
plt.imshow(images[1])

plt.figure()
plt.imshow(images_aug[0])
plt.figure()
plt.imshow(images_aug[1])
plt.show()
