import cv2
import numpy as np
import sys


# 轮廓定位
def contourSegmentation(frame,src):

    image,contours,hierarchy = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    imag = cv2.drawContours(frame, contours, 3, (0,255,0), 3)

    n = len(contours)       # 轮廓个数
    # print(n)
    dst = frame
    for i in range(n):
        # length = cv2.arcLength(contours[i], True)      # 获取轮廓长度
        area = cv2.contourArea(contours[i])             # 获取轮廓面积
        # print('length['+str(i)+']长度=',length)
        # print("contours["+str(i)+"]面积=",area)
        if area < 300:
            continue 
        x, y, w, h = cv2.boundingRect(contours[i])  

        if w>h*1.3 or h>w*1.3:
            continue

        cut = frame[y:y+h,x:x+w]
        img = cv2.rectangle(src,(x,y),(x+w,y+h),(0,255,0),4)
        # cv2.imshow('img',img)
        # 二值取反
        # dst = access_pixels(cut)
        # cv2.imshow('acc',cut)
        dst = cv2.resize(dst,(300,300))
    
    return dst,imag



if __name__ == "__main__":
    # Read video 默认摄像头流       # RTSP 拉流
    video = cv2.VideoCapture(0)    
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
        src = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        ret,src = cv2.threshold(src,100,127,0)
        if ret > 0:
            dst ,imag = contourSegmentation(src,frame)
            # cv2.imshow('dst',dst)
            cv2.imshow('src',src)
        cv2.imshow('test',frame)
        k = cv2.waitKey(50) & 0xff
        if k == 27 : 
            break