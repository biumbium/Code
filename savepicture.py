# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:30:58 2019

@author: 15218
"""


#coding: utf-8
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
  
# 声明图片宽高
rows = 28
cols = 28
  
# 要提取的图片数量
images_to_extract = 100
  
# 当前路径下的保存目录
save_dir = "./mnist_digits_images"
  
# 读入mnist数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
imgs_train, labels_train = mnist.train.images, mnist.train.labels
for i in range(10):
    if not os.path.exists(str(i)):
        os.makedirs(str(i))
cnt = [0 for i in range(10)]
for i in range(imgs_train.shape[0]):
    array = (imgs_train[i].reshape((28, 28)) * 255).astype(np.uint8)
    cnt[labels_train[i]] += 1
    img = Image.fromarray(array, 'L')
    img.save(str(labels_train[i]) + '\\' + str(cnt[labels_train[i]]) + '.jpg')