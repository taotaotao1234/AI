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

train_idx = 0
npy_idx = 0
image_size = 28
# path = '../dataset/'
# path = '../data/'
# path = '../data2/'
path = '../data4/'

files = os.listdir(path)
random.shuffle(files)
images = []
labels = []
for f in files: #目录下所有文件夹
    file_path = os.path.join(path, str(f)) + '/'
    for root, dirs, files in os.walk(file_path):
        for file in files:
            # if os.path.splitext(file)[1] == '.bmp':
            if os.path.splitext(file)[1] == '.png':
                train_idx = train_idx + 1
                img_path = os.path.join(file_path, str(file))
                # print('img_path={}'.format(img_path))
                # img = image.load_img(img_path)
                # img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                # img = cv2.resize(img, (28, 28))
                # img = cv2.imread(img_path)
                img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (28, 28))
                # img = img.reshape((-1, 28, 28, 1)).astype('float') 
                # img_array = image.img_to_array(img)
                img_array = np.asarray(img)
                images.append(img_array)
                labels.append(f) 
                if train_idx % 500 == 0:
                    print('read {}'.format(train_idx))

images = np.array(images)   #（num, h, w, 3）
labels = np.array(labels)   #(num, )
if labels[0] != 0 :
    print("success,{},{}".format(labels[0],train_idx))
# images /= 255

# 490个数据，每个数字49
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)  #划分训练数据、训练标签、验证数据、验证标签
print(images.shape, images.shape)
# print(images[0])
print(labels[0])
# plt.imshow(images[0])
# plt.show()

# 搭建LeNet网络
def LeNet():
    network = models.Sequential()
    network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    network.add(layers.AveragePooling2D((2, 2)))
    network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    network.add(layers.AveragePooling2D((2, 2)))
    network.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
    network.add(layers.Flatten())
    network.add(layers.Dense(84, activation='relu'))
    # 结果分类
    # network.add(layers.Dense(10, activation='softmax'))
    network.add(layers.Dense(13, activation='softmax'))
    return network
network = LeNet()
#network =  load_model('model.h5') 
# 编译
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
images = images.reshape((images.shape[0], 28, 28, 1)).astype('float') / 255
labels = to_categorical(labels)
# 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
# network.fit(images, labels, epochs=10, batch_size=128, verbose=2)
network.fit(images, labels, epochs=20, batch_size=128, verbose=2)

# save model
# mp = "./model.h5"
# mp = "./model_simple.h5"
mp = './my_model.h5'
network.save(mp)


for a in range(10):
    # path = '../pic/000.png'
    path = '../pic/{}{}{}.png'.format(a,a,a)
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    ret,img = cv2.threshold(img, 200, 255, 0)
    # print(path)
    img = cv2.resize(img, (28, 28))
    img = img.reshape((-1, 28, 28, 1)).astype('float') / 255
    #img = np.resize(img,new_shape=(1,784))
    lasses  = network.predict(img)
    # print(lasses)

    temp = lasses[0][0]
    num = 0
    for i in range(13):
        if temp < lasses[0][i]:
            temp = lasses[0][i]
            num = i

    print('a = {}; num = {}; probability = {}'.format(a,num,lasses[0][num]))




