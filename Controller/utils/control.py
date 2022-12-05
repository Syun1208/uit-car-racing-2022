import numpy as np
import cv2
import time
import math

#   Hàm dự đoán làn đường dựa vào ảnh từ camera
def road_lines(image, session, inputname):
	# Crop ảnh lại, lấy phần ảnh có làn đường
	image = image[200:, :, :]
	small_img = cv2.resize(image, (160, 80))
	small_img = np.array(small_img, dtype=np.float32)
	small_img = small_img[None, :, :, :]
	prediction = session.run(None, {inputname: small_img})
	prediction = np.squeeze(prediction)
	prediction = np.where(prediction < 0.5, 0, 255)
	prediction = prediction.reshape(80, 160)
	prediction = prediction.astype(np.uint8)
	return cv2.cvtColor(prediction,cv2.COLOR_GRAY2RGB)
