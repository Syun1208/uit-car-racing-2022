from PIL import Image
import math
import os
import time
import argparse
import numpy as np
import onnxruntime
# import psutil
import sys
import time
import cv2
import pycuda.autoinit
from deploy.controller import *
from utils.yolo_classes import get_cls_dict
from utils.display import show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.control import road_lines
from imutils.video import VideoStream
#sess_options = onnxruntime.SessionOptions()
import UITCar
import time
DETEC = True
SHOW_IMG = True
PRINT = False
session_lane = onnxruntime.InferenceSession(
    'new_lane.onnx', None, providers=['CPUExecutionProvider'])
input_name_lane = session_lane.get_inputs()[0].name
session_sign = onnxruntime.InferenceSession(
    'sign_new.onnx', None, providers=['CUDAExecutionProvider'])
input_name_sign = session_sign.get_inputs()[0].name


def gstreamer_pipeline(
    capture_width=640,
    capture_height=360,
    display_width=640,
    display_height=360,
    framerate=20,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor_mode=0 !"
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
# import copy


def remove_small_contours(image):
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
    return image_remove

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    #model yolo mặc định 6 class
    parser.add_argument(
        '-c', '--category_num', type=int, default=5,
        help='number of object categories [5]')
    #chọn model
    parser.add_argument(
        '-m', '--model', type=str, default='update-416',
        help=('model object detection'))
    args = parser.parse_args()
    return args

def set_angle(speed, angle):
    if abs(angle) > 44:
        if angle > 44:
            angle = 44
        elif angle < -44:
            angle = 44
    return speed, angle


def loop_and_detect(cam, trt_yolo, conf_th, vis):

    fps = 0.02
    Car = UITCar.UITCar()
    Car.setMotorMode(0)
    Car.setAngle(0)
    Car.setSpeed_rad(0)
    area = 0
    cl = -1
    start_time_detect = time.time()
    maxSpeed = 20
    imgs = os.listdir('warnup')
    for img in imgs:
        img = cv2.imread(os.path.join('warnup', img))
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        _ = vis.draw_bboxes(
            img, boxes, confs, clss, session_sign, input_name_sign)
    print("done warnup")
    i = 0
    # if Car.button == 1:
    start = time.time()
    flag = False
    while not flag:
        Car.setSpeed_rad(maxSpeed)
        if time.time() - start > 3:
            flag = True
    while True:
        tic = time.time()
        img = cam.read()  # đọc ảnh từ camera
        cv2.imshow('Img', img)
        if img is None:
            print("None")
            break
        # try:
        image_segmentation= road_lines(np.copy(img), session=session_lane, inputname=input_name_lane)  # hàm segment làn đường trả về ảnh đã segment
        image_segmentation = remove_small_contours(image_segmentation)
        cv2.imshow('Mask', image_segmentation)
        # trả về boxes: chứa tọa độ bounding box, phần trăm dự đoán, và phân lớp
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        area, cl = vis.draw_bboxes(
        img, boxes, confs, clss, session_sign, input_name_sign)
        '''-------------------------Controller-----------------------'''
        controller = Controller(image_segmentation, maxSpeed, cl, Car, area, confs)
        angle, speed = controller()   
        speed, angle = set_angle(speed, angle)   
        Car.setSpeed_rad(speed)
        Car.setAngle(angle)
        # print('FPS: ',  1.0 /(time.time() - tic))
        # print('Speed: ', speed)
        # print('Angle: ', angle)
        Car.OLED_Print('Speed: {}'.format(speed), 1)
        Car.OLED_Print('Angle: {}'.format(angle), 2)
        Car.OLED_Print('BBox Area: {}'.format(area), 3)
        Car.OLED_Print('Confidence: {}'.format(confs), 4)
        '''-------------------------Controller-----------------------'''
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
    # else:
    #     Car.setSpeed_rad(0)
    #     Car.setAngle(0)
    # else:
    #     Car.setSpeed_rad(0)
    #     Car.setAngle(0)
        # if SHOW_IMG:
        #     # hàm vẽ bounding box lên ảnh

        #     toc = time.time()
        #     fps = 1.0 / (toc - tic)
        #     img = show_fps(img, fps)
        #     img[:image_segmentation.shape[0], img.shape[1] -
        #         image_segmentation.shape[1]:, :] = image_segmentation
        #     cv2.imshow('merge', img)

        # except Exception as inst:
        #     print(type(inst))    # the exception instance
        #     print(inst)
        #     pass


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    print("---------------------------------------------")

    cam = VideoStream(src=gstreamer_pipeline()).start()  # đọc camera
    print("********************************************")
    cls_dict = get_cls_dict(args.category_num)  # lấy danh sách tên các lớp
    #---------------------------Kiểm tra model có bị lỗi không để lấy kích thước ảnh
    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
    #---------------------------------------------------------------------------------------
    print("===============================================")
    trt_yolo = TrtYOLO(args.model, (h, w),
                       args.category_num)  # load model yolo
    print("++++++++++++++++++Read model done+++++++++++++++++")
    # gọi instance BBoxVisualization với thông số mặc định là danh sách các lớp
    vis = BBoxVisualization(cls_dict)
    # m=None
    # vào vòng lặp để dự đoán liên tục
    loop_and_detect(cam, trt_yolo, conf_th=0.7, vis=vis)

    cam.stream.release()  # release camera
    cam.stop()
    cv2.destroyAllWindows()
    #tắt hết các windows của images


if __name__ == '__main__':
    main()
