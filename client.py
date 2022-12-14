
from PIL import Image
import math

from PIL import Image
import math

import os
import time
import argparse
import numpy as np
import onnxruntime
import sys
import time
import cv2
import pycuda.autoinit  
from utils.yolo_classes import get_cls_dict, COCO_CLASSES_LIST

from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
# from utils.control import road_lines
from imutils.video import VideoStream
import time
# sess_options = onnxruntime.SessionOptions()

### Client ###
from controller import Controller, road_lines, remove_small_contours

Control = Controller(1, 0.05)

SSH = False
DETEC = True
SHOW_IMG = True
PRINT = False
session_lane = onnxruntime.InferenceSession( 'weights/lane_mod.onnx', None, providers=['CPUExecutionProvider'])
input_name_lane = session_lane.get_inputs()[0].name
session_sign = onnxruntime.InferenceSession('weights/sign_new.onnx', None, providers=['CUDAExecutionProvider'])
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

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    #model yolo m???c ?????nh 6 class
    parser.add_argument(
        '-c', '--category_num', type=int, default=5,
        help='number of object categories [6]')
    #ch???n model
    parser.add_argument(
        '-m', '--model', type=str, default='update-416',
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis):

    # be tap lam quen :D
    img = cv2.imread("betaplamquen.jpg")
    for i in range(10):
        boxes, confs, clss = trt_yolo.detect(img, conf_th)              #tr??? v??? boxes: ch???a t???a ????? bounding box, ph???n tr??m d??? ??o??n, v?? ph??n l???p
        
        img, class_TS, area = vis.draw_bboxes(
            img, boxes, confs, clss, session=session_sign, inputname=input_name_sign)
        print("taplamquen", class_TS)

            
    # fps = 0.02
    # start_time_detect=time.time()
    frame = 0

    lst_area = np.zeros(5)

    while True:
        tic = time.time()
        img = cam.read()                    #?????c ???nh t??? camera
        if img is None:
            print("None")
            break
        # try:        
        copyImage = np.copy(img) 

        #### Segmentation ####           
        DAsegmentation = road_lines(copyImage, session=session_lane, inputname=input_name_lane)    #h??m detect l??n ???????ng tr??? v??? ???nh ???? detect, g??c l??i, v???i ??i???m center d??? ??o??n      
        DAsegmentation = remove_small_contours(DAsegmentation)

        #### Object Detection ####
        # if frame % 2 == 0:
        boxes, confs, clss = trt_yolo.detect(img, conf_th)              #tr??? v??? boxes: ch???a t???a ????? bounding box, ph???n tr??m d??? ??o??n, v?? ph??n l???p
        if(len(confs)>0):                                               #n???u nh???n di???n ???????c bi???n b??o
            #ch??? l???y ra ?????i t?????ng c?? ph???n tr??m d??? ??o??n cao nh???t
            index=np.argmax(confs)
            confs = [confs[index]]
            
            #### Class Name ####
            clss = [clss[index]]

        # h??m v??? bounding box l??n ???nh, tr??? v??? ???nh ???? v???, center c???a c??c ?????i t?????ng, ph??n l???p, v?? di???n t??ch c???a boundingbox
        img, class_TS, area = vis.draw_bboxes(
            img, boxes, confs, clss, session=session_sign, inputname=input_name_sign)

        lst_area[1:] = lst_area[0:-1]
        lst_area[0] = area
        ############# Control Car ############# 
        pred = np.copy(DAsegmentation)

        sendBack_angle, sendBack_speed = Control(pred, sendBack_speed=28, height=10, signal=class_TS, area=area)

        #####------Show h??nh-----------------------
        # if SHOW_IMG:
            # img[:DAsegmentation.shape[0],img.shape[1]-DAsegmentation.shape[1]:,:]=DAsegmentation
            # cv2.imshow('merge',img)

        frame += 1
        # toc = time.time()
        # fps = 1.0 / (toc - tic)
        # print(fps)
        # key = cv2.waitKey(1)
        # if key == 27:  # ESC key: quit program
        #     break
            
        # except Exception as inst:
        #     print(type(inst))    # the exception instance
        #     print(inst)
        #     pass
    Car.setAngle(0)
    Car.setSpeed_cm(0)
    # del Car

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    print("---------------------------------------------")

    cam = VideoStream(src=gstreamer_pipeline()).start()            #?????c camera
    print("********************************************")
    cls_dict = get_cls_dict(args.category_num)                                  #l???y danh s??ch t??n c??c l???p   
    print(cls_dict, args.category_num)
    #---------------------------Ki???m tra model c?? b??? l???i kh??ng ????? l???y k??ch th?????c ???nh                            
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
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)                   #load model yolo
    print("++++++++++++++++++Read model done+++++++++++++++++")
    vis = BBoxVisualization(cls_dict)                                           #g???i instance BBoxVisualization v???i th??ng s??? m???c ?????nh l?? danh s??ch c??c l???p
    # m=None
    loop_and_detect(cam, trt_yolo, conf_th=0.5, vis=vis)                        #v??o v??ng l???p ????? d??? ??o??n li??n t???c
    
    cam.stream.release()                                                        #release camera
    cam.stop()
    cv2.destroyAllWindows()   
                                                      #t???t h???t c??c windows c???a images

if __name__ == '__main__':
    main()