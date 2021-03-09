from PIL import Image
import cv2
import numpy as np
import os.path as osp
import sys
from yolo import YOLO
from ROOTNET.demo import demo
from POSENET.demo import demo1
yolo = YOLO()
cam = cv2.VideoCapture('ball_test01.mp4')
# cam = cv2.VideoCapture(0)
width = 1920  # 定义摄像头获取图像宽度
height = 1080   # 定义摄像头获取图像长度

cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)  #设置宽度
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  #设置长度
num = 0
sz = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
      int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = 15
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vout_1 = cv2.VideoWriter()
vout_1.open('./test_vedio.mp4', fourcc, fps, sz, True)
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame,(372, 495), interpolation=cv2.INTER_CUBIC)
    try:

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        r_image, bbox_list = yolo.detect_image(image)
        img = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
        for i in range(len(bbox_list)):
            x, y = bbox_list[i][1], bbox_list[i][0]
            bbox_list[i][1], bbox_list[i][0] = y, x
            x, y = bbox_list[i][2], bbox_list[i][3]
            bbox_list[i][2], bbox_list[i][3] = y-bbox_list[i][0], x-bbox_list[i][1]
        root_list = demo.rootnet(frame, bbox_list)
        pose_img = demo1.posenet(img, bbox_list, root_list, num)
        pose_img = cv2.cvtColor(np.asarray(pose_img), cv2.COLOR_RGB2BGR)
        pose_img = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)
        vout_1.write(pose_img)
        num += 1

        if cv2.waitKey(1) & 0xFF == ord("w"):
            vout_1.release()
            break
    except:
        print('can not detect the people')
        continue

