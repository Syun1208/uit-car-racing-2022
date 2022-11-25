import onnxruntime
import cv2
import numpy as np
import time
import pycuda.autoinit  
from utils.yolo_with_plugins import TrtYOLO
# from tensorflow.keras.models import load_model
# assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
# sess_options.optimized_model_filepath = ('lane_gpu.onnx')


session1 = onnxruntime.InferenceSession(
	'lane.onnx', None, providers=['CPUExecutionProvider'])
input_name1 = session1.get_inputs()[0].name
input_shape1 = session1.get_inputs()[0].shape

session2 = onnxruntime.InferenceSession('sign1.onnx',None,providers=['CUDAExecutionProvider'])
input_name2 = session2.get_inputs()[0].name
input_shape2 = session2.get_inputs()[0].shape

trt_yolo = TrtYOLO('yolov4-tiny-416', (416, 416), 6)                   #load model yolo
image = cv2.imread('demo5.jpg')
def pre_proc(image):
	image = image[200:, :, :]

	 # Resize ảnh lại = với kích thước input của model
	small_img = cv2.resize(image, (160, 80))
	 # cv2.imshow("crop",small_img)
	small_img = np.array(small_img)
	small_img = small_img[None, :, :, :]
	return small_img
def road_lines(image,session,inputname):
	 # Crop ảnh lại, lấy phần ảnh có làn đường
	small_img = cv2.resize(image, (160, 80))
	 # cv2.imshow("crop",small_img)
	small_img = np.array(small_img,dtype=np.float32)
	small_img = small_img[None, :, :, :]
	prediction = session.run(None,{inputname:small_img})
	prediction=np.squeeze(prediction)
	# prediction = prediction*255
	prediction = np.where(prediction < 0.5, 0, 255)
	prediction = prediction.reshape(80, 160)
	 # print(prediction.shape)
	prediction = prediction.astype(np.uint8)
	
	return prediction
def sign1_proc(image,session,inputname):
    pred = session.run(None,{inputname:image})
    return pred
sign1_feed= np.random.rand(1,30,30,3).astype('f')
runtime =[]
for i in range(100):
	st = time.time()
	road_lines(image,session1,input_name1)
	trt_yolo.detect(image,0.7)
	# sign1_proc(sign1_feed,session2,input_name2)
	
	runtime.append(1/(time.time()-st))
print(np.mean(runtime))
# out = sign1_proc(sign1_feed,session2,input_name2)
# out = np.squeeze(out)
# print(out)
