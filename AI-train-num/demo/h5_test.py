import os
import cv2
import numpy as np
from keras.models import load_model

# 二值取反
def access_pixels(image):
    height, width = image.shape
    for row in range(height):
        for col in range(width):
            pv = image[row,col]
            image[row,col] = 255 - pv
    return image

# 形态学变换
def  morphologyChange(frame,data):
    kernel = np.ones((data,data),np.uint8)

    # 腐蚀（去噪）
    # erosion = cv2.erode(frame,kernel,iterations = 1)
    # 膨胀（联通相邻区域）
    # dilation = cv2.dilate(erosion,kernel,iterations = 1)

    # 开运算（先腐蚀再膨胀）
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    # 闭运算（先膨胀再腐蚀）
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
"""---------载入已经训练好的模型---------"""
# new_model = load_model('model.h5')
# new_model = load_model('model_simple.h5')
new_model = load_model('my_model.h5')

"""---------用opencv载入一张待测图片-----"""
path = '../pic/70.png'
img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
ret,src = cv2.threshold(img, 200, 255, 0)
cv2.imshow("test picture", src)

dst = cv2.resize(src, (28, 28))
dst=dst.astype(np.float32)

# 将灰度图转化为1*784的能够输入的网络的数组
#picture=1-dst/255
#picture=np.reshape(picture,(1,28,28,1))
picture = dst
picture=np.reshape(picture,(-1, 28, 28, 1)).astype('float') / 255

# cv2.imshow("picture", picture)
# 用模型进行预测
y = new_model.predict(picture)
print("softmax:")
for i,prob in enumerate(y[0]):
    print("class{},Prob:{}".format(i,prob))
result = np.argmax(y)
print("你写的数字是：", result)
print("对应的概率是：",np.max(y[0]))
cv2.waitKey(20170731)

'''
# 载入图片
for a in range(10):
    # path = '../pic/70.png'
    path = '../pic/{}{}{}.png'.format(a,a,a)

    src = cv2.imread(path)
    cv2.imshow("test picture", src)

    lower_red = np.array([156,205,131])
    upper_red = np.array([180,228,255])
    src=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    src = cv2.inRange(src,lower_red,upper_red)
    src = access_pixels(src)
    cv2.imshow('mask',src)

    blur = cv2.blur(src,(3,3))
    cv2.imshow('blur',blur)
    GaussianBlur = cv2.GaussianBlur(src,(3,3),0)
    cv2.imshow('GaussianBlur',GaussianBlur)
    medianBlur = cv2.medianBlur(src,3)
    cv2.imshow('medianBlur',medianBlur)


    src = morphologyChange(medianBlur,1)
    cv2.imshow('morphologyChange',src)

    # 将图片转化为28*28的灰度图
    # src = cv2.cvtColor(src, cv2.COLOR_HSV2BGR)
    # src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    dst = cv2.resize(src, (28, 28))
    dst=dst.astype(np.float32)
    
    # 将灰度图转化为1*784的能够输入的网络的数组
    #picture=1-dst/255
    #picture=np.reshape(picture,(1,28,28,1))
    picture = dst
    picture=np.reshape(picture,(-1, 28, 28, 1)).astype('float') / 255

    # 用模型进行预测
    y = new_model.predict(picture)
    print("softmax:")
    for i,prob in enumerate(y[0]):
        print("class{},Prob:{}".format(i,prob))
    result = np.argmax(y)
    print("你写的数字是：", result)
    print("对应的概率是：",np.max(y[0]))
    cv2.waitKey(20170731)
'''