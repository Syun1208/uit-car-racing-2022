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
from HarwareControl.UITCar import UITCar
import pycuda.autoinit
from utils.yolo_classes import get_cls_dict, COCO_CLASSES_LIST
from utils.display import show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from deploy.controller import Controller
from utils.control import road_lines
from imutils.video import VideoStream

# sess_options = onnxruntime.SessionOptions()

DETEC = True
SHOW_IMG = True
PRINT = False
Car = UITCar()
session_lane = onnxruntime.InferenceSession('weights/lane_mod.onnx', None, providers=['CPUExecutionProvider'])
input_name_lane = session_lane.get_inputs()[0].name
session_sign = onnxruntime.InferenceSession('sign_new.onnx', None, providers=['CUDAExecutionProvider'])
input_name_sign = session_sign.get_inputs()[0].name


def set_angle_speed(angle, speed):
    Car.setAngle(angle)
    if abs(angle) > 50:  # nếu dự đoán góc trên 50 thì cho bẻ góc tối đa
        if angle > 50:
            angle = 50
        elif angle < -50:
            angle = -50
        Car.setAngle(-angle * 0.8)
    elif abs(angle) < 20:
        Car.setAngle(-angle * 0.7)
    else:
        Car.setAngle(-angle * 0.8)
    Car.setSpeed_rad(speed)


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

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    # model yolo mặc định 6 class
    parser.add_argument(
        '-c', '--category_num', type=int, default=6,
        help='number of object categories [6]')
    # chọn model
    parser.add_argument(
        '-m', '--model', type=str, default='yolo-tiny-6cl-416',
        help='model object detection')
    args = parser.parse_args()
    return args


def BTN1_Func(channel):
    print("Pressed 1")


def BTN2_Func(channel):
    print("Pressed 2")


def BTN3_Func(channel):
    print("Pressed 3")


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    fps = 0.02
    start_time_detect = time.time()
    # Car.regBTN(1, BTN1_Func)
    Car.setMotorMode(0)  # 0: controlled by speed, 1: controlled by distance.
    # Car.setSpeed_rad(20)
    while True:

        tic = time.time()
        img = cam.read()  # đọc ảnh từ camera
        if img is None:
            print("None")
            break
        # try:          
        image_segmentation = road_lines(np.copy(img), session=session_lane,
                                        inputname=input_name_lane)  # hàm segment làn đường trả về ảnh đã segment
        cv2.imshow('Segmentation', image_segmentation)
        boxes, confs, clss = trt_yolo.detect(img,
                                             conf_th)  # trả về boxes: chứa tọa độ bounding box, phần trăm dự đoán, và phân lớp
        if len(confs) > 0.5:  # nếu nhận diện được biển báo
            # chỉ lấy ra đối tượng có phần trăm dự đoán cao nhất
            index = np.argmax(confs)
            confs = [confs[index]]
            classDetect = int(clss[index])
            clss = [clss[index]]
        img, centers, class_TS, area = vis.draw_bboxes(
            img, boxes, confs, clss, session=session_sign, inputname=input_name_sign)
        if len(centers) > 0:  # nếu phát hiện đối tượng
            # phải sau 2s so với lần xử lý biển báo trước thì mới cho xử lý biển báo tiếp theo
            if time.time() - start_time_detect > 2 and area > 2400:
                if class_TS == COCO_CLASSES_LIST.index('trai'):
                    print('trai')
                elif class_TS == COCO_CLASSES_LIST.index('thang'):
                    print('thang')
                start_time_detect = time.time()
        controller = Controller(image_segmentation, class_TS, Car.getSpeed_rad())
        angle, speed, trafficSigns = controller()
        Car.OLED_Print('------------------------------------', str(speed), 3)
        Car.OLED_Print('Angle: ', str(angle), 4)
        Car.OLED_Print('Speed: ', str(speed), 5)
        Car.OLED_Print('Sign: ', str(trafficSigns), 6)
        Car.OLED_Print('------------------------------------', str(speed), 7)
        set_angle_speed(angle, speed)
        if SHOW_IMG:
            # hàm vẽ bounding box lên ảnh
            img = vis.draw_bboxes(img, boxes, confs, clss)
            toc = time.time()
            fps = 1.0 / (toc - tic)
            img = show_fps(img, fps)
            img[:image_segmentation.shape[0], img.shape[1] - image_segmentation.shape[1]:, :] = image_segmentation
            cv2.imshow('merge', img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break

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
    # ---------------------------Kiểm tra model có bị lỗi không để lấy kích thước ảnh                            
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
    # ---------------------------------------------------------------------------------------
    print("===============================================")
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)  # load model yolo
    print("++++++++++++++++++Read model done+++++++++++++++++")
    vis = BBoxVisualization(cls_dict)  # gọi instance BBoxVisualization với thông số mặc định là danh sách các lớp
    # m=None
    '''----------------------------Controller--------------------------------------------'''
    loop_and_detect(cam, trt_yolo, conf_th=0.7, vis=vis)  # vào vòng lặp để dự đoán liên tục

    cam.stream.release()  # release camera
    cam.stop()
    cv2.destroyAllWindows()
    # tắt hết các windows của images


if __name__ == '__main__':
    main()
