# _*_ coding: utf-8 _*_

from keras.utils import to_categorical
from keras import models, layers
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.models import load_model 
import cv2
import numpy as np
import sys
import subprocess
import time

x=0; y=0; w=0; h=0
path1 = './pic/1.png'
image1 = cv2.imread(path1)
path0 = './pic/0.png'
image0 = cv2.imread(path0)
path2 = './pic/2.png'
image2 = cv2.imread(path2)
path3 = './pic/3.png'
image3 = cv2.imread(path3)
path4 = './pic/4.png'
image4 = cv2.imread(path4)
path5 = './pic/5.png'
image5 = cv2.imread(path5)
path6 = './pic/6.png'
image6 = cv2.imread(path6)
path7 = './pic/7.png'
image7 = cv2.imread(path7)
path8 = './pic/8.png'
image8 = cv2.imread(path8)
path9 = './pic/9.png'
image9 = cv2.imread(path9)

# 目标框选区域
selection = None
# 跟踪开始标志
drag_start = None
track_start = False
track_ok = None
tracker = cv2.TrackerMedianFlow_create()
bbox = (0,0,0,0)

# 颜色定位
def colorDetection(frame,mode):
        
    # 转换HSV颜色空间
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    # 设定提取的颜色阈值    Red 156 180 43 255 46 255 
    # lower_red = np.array([156,43,46])
    # upper_red = np.array([180,255,255])

    # test 160 180 168 253 84 255
    lower_red = np.array([156,147,81])
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
        x_point = 0;y_point = 0
        global x,y,w,h
        global image1
        for i in range(n):
            x, y, w, h = cv2.boundingRect(contours[i])   
            # 去除边缘出现半截数字情况
            if y<frame.shape[0]/20 or x<frame.shape[1]/20 or x+h>frame.shape[1]*0.95 or y+w>frame.shape[0]*0.95:
                continue
            # print(frame.shape[1])
            area = cv2.contourArea(contours[i])     
            # 获取轮廓面积，去除面积较小的噪声   
            if area < 400:
                continue   
            # 去除不规则图形干扰
            if w>h*1.3 or h>w*1.3:
                continue

            # 去除热像仪干扰
            # if x < 300:
            #     continue
            # 画图
            cv2.rectangle(frame, (x,y), (x+w,y+h), (153,153,0), 2) 
            
            res_src = hsv[y:y+h,x:x+w]
            # white 0 180 0 30 221 255
            lower_red = np.array([120,18,137])
            upper_red = np.array([180,60,206])
            res = cv2.inRange(res_src,lower_red,upper_red)
            
            dst = frame[int(y+h/10):int(y+h-h/10),int(x+w/10):int(x+w-w/10)]
            img_add = dst
            x_point = x; y_point = y
            # 图像预测
            ok2,img_add = endput(dst,x,y)

            if ok2 == 1:
                # 图像掩摸
                frame[int(y+h/10):int(y+h-h/10),int(x+w/10):int(x+w-w/10)] = img_add
                
            ok = 1
    # 数据返回
        # 如果跟踪成功
 
        return x_point,y_point,dst,ok
    return mask,res

# 鼠标响应函数
def onmouse(event, x, y, flags, param):
    global selection, drag_start, track_window, track_start,tracker
    # 鼠标左键按下
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        track_window = None
    # 开始拖拽
    if drag_start:
        xmin = min(x, drag_start[0])
        ymin = min(y, drag_start[1])
        xmax = max(x, drag_start[0])
        ymax = max(y, drag_start[1])
        selection = (xmin, ymin, xmax, ymax)
    # 鼠标左键弹起
    if event == cv2.EVENT_LBUTTONUP:
        drag_start = None
        selection = None
        track_window = (0,0,0,0)
        # tracker.init(frame,track_window)
        track_window = (xmin, ymin, xmax - xmin, ymax - ymin)
        if track_window and track_window[2] > 0 and track_window[3] > 0:
            track_start = True
            # 跟踪器以鼠标左键弹起时所在帧和框选区域为参数初始化
            
            tracker_temp = cv2.TrackerMedianFlow_create()
            tracker_temp.init(frame, track_window)
            tracker = tracker_temp


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


#轮廓定位
def contourSegmentation(frame):
    image,contours,hierarchy = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)       #轮廓个数
    img2 = frame

    for i in range(n):
        #length = cv2.arcLength(contours[i], True)  #获取轮廓长度
        area = cv2.contourArea(contours[i])        #获取轮廓面积
        #print('length['+str(i)+']长度=',length)
        #print("contours["+str(i)+"]面积=",area)
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
    if temp > 0.99:
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


def endput(dst,x,y):
    global bbox
    h,w,_ = dst.shape
    img_add = dst
    img = cv2.resize(dst, (300,300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,img = cv2.threshold(img, 100, 255, 0)
    img = morphologyChange(img,1)
    # cv2.imshow('dst',img)
    img = cv2.resize(img, (28,28))
    # cv2.imshow('dst',img)
    num,probability = dection(img)
    ok = 0
    if num >= 0:
        mask = image8
        if num == 0:
            mask = image0
        elif num == 1:
            mask = image1
        elif num == 2:
            mask = image2
        elif num == 3:
            mask = image3
        elif num == 4:
            mask = image4
        elif num == 5:
            mask = image5
        elif num == 6:
            mask = image6
        elif num == 7:
            mask = image7
        elif num == 8:
            mask = image8
        elif num == 9:
            mask = image9

        elif num == 10:
            num = 'A'
        elif num == 11:
            num = 'B'
        elif num == 12:
            num = 'C'
        # print('num = {}; probability = {}'.format(num,probability))  
        cv2.putText(frame, str(num), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 1)
    
        imgnew = cv2.resize(mask, (w,h), interpolation=cv2.INTER_AREA)  #两张图片需是大小相同的
        alpha = 0.1  #将两张图片叠加在一起第一张图片的权重
        beta = 1 - alpha  #第二张图片的权重
        gamma = 0  #一个加到权重总和上的标量值,dst = src1*alpha+ src2*beta + gamma;
        img_add = cv2.addWeighted(dst, alpha, imgnew, beta, gamma)
                    
        bbox = (int(x*1.0),int(y*1.0),int(w*1.0),int(h*1.0))

        ok = 1
    return ok,img_add


# 稀疏光流跟随，延长检测结果显示时间
def MedianFlow():
    pass


if __name__ == "__main__":
    # global track_ok,track_start,tracker
    # network = load_model('model.h5') 
    network = load_model('my_model.h5') 
    #network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy']) 

    # Read video 默认摄像头流
    video = cv2.VideoCapture(0)

    # RTSP 拉流
    # video = cv2.VideoCapture("rtsp://admin:DM9000AEP@192.168.199.163/cam/realmonitor?channel=1&subtype=0")

    # cv2.namedWindow('test')
    # # 为窗口绑定鼠标响应函数onmouse
    # cv2.setMouseCallback('test', onmouse)

    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
    
    # # 推流
    rtmp = 'rtmp://localhost:1935/live/test'
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + 'x' + str(size[1])   # 设置视频流像素
    command = ['ffmpeg',
        '-y', '-an',                            # 覆盖输出文件，不输出音频
        '-f', 'rawvideo',
        '-vcodec','rawvideo',                   # 解码时的解码器
        '-pix_fmt', 'bgr24',                    # 解码格式
        '-s', sizeStr,
        '-r', '25',                             # 设置视频流 FPS
        '-i', '-',                              # 文件输入选项
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',                  # 编码格式
        '-preset', 'ultrafast',
        '-f', 'flv',
        # '-b','1000000',
        rtmp]
    
    pipe = subprocess.Popen(command
        , shell=False
        , stdin=subprocess.PIPE
    )


    track_test = 0

    while True:
        timer = cv2.getTickCount()
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # # 以矩形标记鼠标框选区域
        # if selection:
        #     x0, y0, x1, y1 = selection
        #     cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2, 1)

        # if track_test == 1:
        #     if track_start == True:
        #         track_ok = None
        #         track_ok, box = tracker.update(frame)
                
        #         if track_ok:
        #             p1 = (int(box[0]), int(box[1]))
        #             p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        #             cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        #             print('p1 = {}; p2 = {}'.format(p1,p2))
        #             print("track = {}, box = {},{},{},{}".format(track_ok,int(box[0]),int(box[1]),int(box[2]),int(box[3])))
        #         else:
        #             print("track failed")

        # # 1.颜色定位
        # x,y,dst,ok = colorDetection(frame,2)

        # if track_test == 1:
        # # 更新跟踪器得到最新目标区域
        #     if ok == 1:
        #         if bbox and bbox[2]>0 and bbox[3]>0 :
        #             track_start = True
        #             tracker_temp = cv2.TrackerMedianFlow_create()
        #             bbox = (500,500,50,50)
        #             tracker_temp.init(frame, bbox)
        #             tracker = tracker_temp
        #             print(bbox)       

        cv2.imshow('test',frame)
        pipe.stdin.write(frame.tostring())
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS : " + str(int(fps)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        k = cv2.waitKey(20) & 0xff
        if k == 27 : break
        # time.sleep(0.1)

    video.release()
    pipe.terminate()


