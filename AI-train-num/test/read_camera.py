# _*_ coding: utf-8 _*_

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
    # cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask)
    # cv2.imshow('res',res)
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
            dst = frame[int(y+h/10):int(y+h-h/10),int(x+w/10):int(x+w-w/10)]
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
        kernel = np.ones((15,15),np.uint8)
        frame = cv2.dilate(frame,kernel,iterations = 1)
    if mode == 2:
        kernel = np.ones((3,3),np.uint8)
        # 开运算（先腐蚀再膨胀）
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        # 闭运算（先膨胀再腐蚀）
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    return frame


if __name__ == "__main__":
    # Read video 默认摄像头流
    # video = cv2.VideoCapture(0)
    # RTSP 拉流
    # video = cv2.VideoCapture("rtsp://admin:DM9000AEP@192.168.199.241/cam/realmonitor?channel=1&subtype=0")
    video = cv2.VideoCapture("rtmp://localhost:1935/live/test")
    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
    count = 0
    while True:
        
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # # 1.颜色定位,图片分割及处理
        # mask,res,dst,ok = colorDetection(frame,2)
        # if ok == 1:
        #     img = cv2.resize(dst, (300,300))
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
        #     ret,img = cv2.threshold(img, 150, 255, 0)
        #     img = morphologyChange(img,1)
        #     img = cv2.resize(img, (28,28))
        #     cv2.imshow('dst',img)
        #     path = './out/{}.png'.format(count)
        #     cv2.imwrite(path,img)
        #     count = count +1
        #     if count % 1000 == 0:
        #         print(count)

        cv2.imshow('test1',frame)
        k = cv2.waitKey(50) & 0xff
        if k == 27 : break

