# Data-augmentation-using-GAN
In emotion classification task

This is a Tensorflow implementation of my paper 
["Emotion Classification with Data Augmentation Using Generative Adversarial Networks"](https://arxiv.org/pdf/1711.00648.pdf), 
which has been accepted by PAKDD2018. <br>

## Modes & Structures
We use CycleGAN as our GAN model <br>
![](http://github.com/SharonZhu/Data-augmentation-using-GAN/raw/master/figure/framework.jpg)

## Datasets
### [SFEW](https://cs.anu.edu.au/few/)
### [JAFFE](http://www.kasrl.org/jaffe.html) 
### [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
Also, you can download it by [Baiduyun](https://pan.baidu.com/s/1RHU_V_mAzRPpz88xJHtC0A) with key: gvcn <br>

## Codes
### CycleGAN
Networks, training, utils related to CycleGAN
### Data utils
Some utils for processin data and toy experiments mentioned in paper
### Emotion
CNN network and training for emotion classification, and the embedding visualization
### Figure
Framework, experiment result and manifold visualization in paper
