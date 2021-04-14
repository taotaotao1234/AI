#!/usr/lib/python3
# -*- coding:UTF-8 -*-
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import cv2
import numpy as np

# 配置参数
ap = argparse.ArgumentParser()  # -v videos/nascar.mp4 不要少了文件夹名字
ap.add_argument("-v", "--video", type=str,
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="medianflow",  # 这个感觉比kcf好
                help="OpenCV object tracker type")
args = vars(ap.parse_args())

# opencv已经实现了的追踪算法，传统算法
# 深度学习？？准确高，实时低，电子设备中难，因为网络大
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,  # 流行，cf相关滤波，14年
    # cf找到待追踪的圆（第一帧所在的位置），再对边缘进行padding，训练一个分类器（或者滤波矩阵），
    # 生成正样本和负样本训练，训练的目标是找到滤波矩阵，计算下一步哪个位置得到响应大。
    # 具体内容，看论文。
    # kcf，改进了计算，提高计算速度。1正负样本选择上改进；2核函数，低维变成高维。
    "boosting": cv2.TrackerBoosting_create,  # 十几年前
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create  # 10年
}
# 速度（实时）和准确性

# 实例化OpenCV's multi-object tracker
# trackers = cv2.MultiTracker_create()  # 实例化多目标追踪。
# tracker = cv2.TrackerCSRT_create()  # 追踪器不要在这里创建，这里创建就只产生一个追踪器
#vs = cv2.VideoCapture(args["video"])  # 这个不要在while里面创建
# if a
#  
# if not args.get("video",False):
#     print("[INFO] starting video stream...")
#     vs = VideoStream(src=0).start()
#     time.sleep(1.0)

# # otherwise
# else:
#     vs = cv2.VideoCapture(args["video"])

# trackers = cv2.MultiTracker_create()
trackers = cv2.TrackerMedianFlow_create()
# trackers = cv2.TrackerCSRT_create()

vs = cv2.VideoCapture(0)
# 视频流
while True:
    # 取当前帧
    frame = vs.read()
    # (true, data)
    frame = frame[1]
    # ret, frame = capture.read()  # 这个就不用cv2.imread(frame)，frame已经是参数了
    # 到头了就结束
    if frame is None:
        break

    # resize每一帧
    (h, w) = frame.shape[:2]  # 原始图片大，计算性能低
    width = 600
    r = width / float(w)
    dim = (width, int(h * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # 追踪结果
    (success, boxes) = trackers.update(frame)  # 没有添加追踪器，是一个空架子。
    print(success, boxes)  # success返回值一直都是True,
    # boxes返回值为：[[333. 172.  45.  76.]]，# 在未画下框的时候，读取为空。
    # [[278. 166.  36.  62.]
    #  [501. 242.  41.  67.]]
    print("trackers：", trackers)
    # 绘制区域
    for box in boxes:  # 确保适合多追踪器
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示
    cv2.imshow("Frame", frame)
    #print(cv2.waitKey(100))  # 返回值为-1，-1&0xFF的结果为255
    key = cv2.waitKey(100) & 0xFF
    # 0xFF是十六进制常数，二进制值为11111111。通过使用位和（和）这个常数，
    # 它只留下原始的最后8位（在这种情况下，无论CV2.WaITKEY（0）是），此处是防止BUG。

    if key == ord("s"):  # ord
        # 选择一个区域，按s
        box = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=False)  # fromCenter选中的为中心点拉。showCrosshair展示中心十字架。
        # 对应显示窗口名称和窗口显示的内容，cv2.imshow("Frame", frame)

        # 创建一个新的追踪器
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()  # 指定追踪器
        trackers.add(tracker, frame, box)  # 添加追踪器，frame哪幅图像，哪个区域box

    # 退出
    elif key == 27:
        break
vs.release()
cv2.destroyAllWindows()
