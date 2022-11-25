"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import numpy as np

import time
# from scipy.misc import imresize
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
from keras.models import load_model
# from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
# from utils.visualization import BBoxVisualization
# from utils.control import carControl
from keras.models import load_model
# from utils.yolo_with_plugins import TrtYOLO


WINDOW_NAME = 'TrtYOLODemo'
CHECKPOINT = 40
IMAGESHAPE = [160, 80]
LANEWIGHT = 90

def carControl(image):
    _lineRow = image[CHECKPOINT, :]
    count = 0
    sumCenter = 0

    for x, y in enumerate(_lineRow):
        if y == 255:
            count += 1
            sumCenter += x
    
    centerArg = int(sumCenter/count)

    if (count < LANEWIGHT):
        if (center < int(IMAGESHAPE[0]/2)):
            center -= LANEWIGHT - count
        else :
            center += LANEWIGHT - count

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    _steering = math.degrees(math.atan((center - int(IMAGESHAPE[0]/2))/int(IMAGESHAPE[1]/2)))
    return img, _steering
def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)

    args = parser.parse_args()
    return args


def loop_and_detect(cam,  model):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        
        image = cam.read()
        if image is None:
            break
        try:
            image=cv2.resize(image,(640,360))
            
            image=image[130:290,:,:]
            
            small_img = cv2.resize(image, (160, 80))
            
            small_img = np.array(small_img)
            small_img = small_img[None,:,:,:]

            prediction = model.predict(small_img)[0] * 255
            prediction=np.where(prediction < 127, 0, 255)

            prediction=prediction.reshape(80,160)

            img = prediction.astype(np.uint8)

            img, angle = carControl(img)

            # print(angle)

            img = show_fps(img, fps)
            cv2.imshow("es", img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
            key = cv2.waitKey(1)
            if key == 27:  # ESC key: quit program
                break
            
        except:
            break


def main():
    args = parse_args()
    

    cam = Camera(args)
    if not cam.isOpened():
        # cam.release()
        raise SystemExit('ERROR: failed to open camera!')


    m=load_model('mod.h5')
    # trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)

    
    # vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam,  m)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
