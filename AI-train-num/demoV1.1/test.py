# _*_ coding: utf-8 _*_

from keras.utils import to_categorical
from keras import models, layers
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.models import load_model 
import cv2
import numpy as np
import sys


# 颜色定位
def colorDetection(frame,mode):
        
    # 转换HSV颜色空间
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    # 设定提取的颜色阈值    Red 156 180 43 255 46 255 
    lower_red = np.array([156,43,46])
    upper_red = np.array([180,255,255])

    # 根据阈值构建掩模
    mask = cv2.inRange(hsv,lower_red,upper_red)

    # 对原图和掩模进行位运算
    res = cv2.bitwise_and(frame,frame,mask=mask)

    # 显示图像
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    if mode == 2:
        image,contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        n=len(contours) 
        ok = 0
        dst = frame
        for i in range(n):
            x, y, w, h = cv2.boundingRect(contours[i])   
            # if x < 300:
            #     continue
            area = cv2.contourArea(contours[i])        #获取轮廓面积
            if area < 1000:
                continue        
            cv2.rectangle(frame, (x,y), (x+w,y+h), (153,153,0), 5) 

            res_src = hsv[y:y+h,x:x+w]
            # white 0 180 0 30 221 255
            lower_red = np.array([120,18,137])
            upper_red = np.array([180,60,206])
            res = cv2.inRange(res_src,lower_red,upper_red)
            
            dst = frame[int(y+h/20):int(y+h-h/20),int(x+w/20):int(x+w-w/20)]

            ok = 1
    # 数据返回
        return mask,res,dst,ok
    return mask,res


# 形态学变换
def  morphologyChange(frame,mode):
    if mode == 1:
        # 腐蚀（去噪）
        kernel = np.ones((3,3),np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        # frame = cv2.erode(frame,kernel,iterations = 1)
        # 膨胀（联通相邻区域）
        kernel = np.ones((9,9),np.uint8)
        frame = cv2.dilate(frame,kernel,iterations = 1)
    if mode == 2:
        kernel = np.ones((3,3),np.uint8)
        # 开运算（先腐蚀再膨胀）
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        # 闭运算（先膨胀再腐蚀）
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    return frame


#轮廓定位
def contourSegmentation(frame):
    image,contours,hierarchy = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)       # 轮廓个数
    img2 = frame

    for i in range(n):
        # length = cv2.arcLength(contours[i], True)     # 获取轮廓长度
        area = cv2.contourArea(contours[i])             # 获取轮廓面积
        # print('length['+str(i)+']长度=',length)
        # print("contours["+str(i)+"]面积=",area)
        if area < 300:
            continue 
        x, y, w, h = cv2.boundingRect(contours[i])   
        img2 = frame[y:y+h,x:x+w]
        # 二值取反
        img2 = access_pixels(img2)
        # cv2.imshow('acc',img2)
        img2 = cv2.resize(img2,(300,300))
    
    return img2


# 二值取反
def access_pixels(image):
    height, width = image.shape
    for row in range(height):
        for col in range(width):
            pv = image[row,col]
            image[row,col] = 255 - pv
    return image


# 数字识别
def dection(image):
    img = cv2.resize(image, (28, 28))
    img = img.reshape((-1, 28, 28, 1)).astype('float') / 255
    lasses  = network.predict(img)

    temp = lasses[0][0]
    num = 0
    # print(lasses[0])
    # for i in range(10):
    for i in range(13):
        if temp < lasses[0][i]:
            temp = lasses[0][i]
            num = i
    # print(num)
    if temp > 0.1:
        return num,lasses[0][num]
    else:
        return -1,-1


# 颜色量化
def colorQuantization(frame):
    img = frame
    Z = img.reshape((-1,1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))    
    return res2 


# 滤波
def filtering(frame,mode,size):
    if mode == 1:
        # 均值滤波
        res = cv2.blur(frame,(size,size))
    elif mode == 2:
        # 高斯滤波
        res = cv2.GaussianBlur(frame,(size,size),0)
    elif mode == 3:
        # 中值滤波
        res = cv2.medianBlur(frame,size)
    else:
        res = frame
    return res


if __name__ == "__main__":

    # network = load_model('model.h5') 
    network = load_model('my_model.h5') 
    #network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy']) 

    # Read video 默认摄像头流
    video = cv2.VideoCapture(0)

    # RTSP 拉流
    # video = cv2.VideoCapture("rtsp://admin:DM9000AEP@192.168.199.163/cam/realmonitor?channel=1&subtype=0")

    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
    
    while True:
        
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # 1.颜色定位
        mask,res,dst,ok = colorDetection(frame,2)
        if ok == 1:
            # cv2.imshow('mask',mask)

            img = cv2.resize(dst, (300,300))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            
            ret,img = cv2.threshold(img, 150, 255, 0)
            img = morphologyChange(img,1)
            # cv2.imshow('dst',img)


            img = cv2.resize(img, (28,28))
            cv2.imshow('dst',img)

            num,probability = dection(img)
            if num >= 0:
                if num == 10:
                    print('num = A;probability = {}'.format(probability)) 
                elif num == 11:
                    print('num = B;probability = {}'.format(probability)) 
                elif num == 12:
                    print('num = C;probability = {}'.format(probability)) 
                else :
                    print('num = {};probability = {}'.format(num,probability))  
            
        cv2.imshow('test',frame)
        k = cv2.waitKey(20) & 0xff
        if k == 27 : break


