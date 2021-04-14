# _*_ coding: utf-8 _*_

from keras.utils import to_categorical
from keras import models, layers
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.models import load_model 
from keras.preprocessing import image
from PIL import ImageFile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import cv2
import numpy as np
import sys
import os
import random
import imageio

# img_path = '../pic/000.png'
# img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (300,300))
# ret,img = cv2.threshold(img, 200, 255, 0)
# img = cv2.resize(img, (28,28))
# cv2.imwrite('../pic/00.png',img)

train_idx = 0
npy_idx = 0
image_size = 28
# path = '../dataset/'
# path = '../data/'
path = '../data2/'
output = '../data3/'
files = os.listdir(path)
random.shuffle(files)

for f in files: #目录下所有文件夹
    file_path = os.path.join(path, str(f)) + '/'
    for root, dirs, files in os.walk(file_path):
        for file in files:
            # if os.path.splitext(file)[1] == '.bmp':
            if os.path.splitext(file)[1] == '.png':
                train_idx = train_idx + 1
                img_path = os.path.join(file_path, str(file))
                img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                ret,img = cv2.threshold(img, 150, 255, 0)
                img = cv2.resize(img, (28, 28))
                cv2.imwrite('{}{}/{}.png'.format(output,f,train_idx),img)
                if train_idx % 500 == 0:
                    print('read {}'.format(train_idx))