import numpy as np
import cv2
import time
import math


#   Hàm dự đoán làn đường dựa vào ảnh từ camera
def road_lines(image, session, inputname):
    # Crop ảnh lại, lấy phần ảnh có làn đường
    image = image[200:, :, :]
    small_img = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    small_img = small_img / 255
    small_img = np.array(small_img, dtype=np.float32)
    small_img = small_img[None, :, :, :]
    prediction = session.run(None, {inputname: small_img})
    prediction = np.squeeze(prediction)
    prediction = np.where(prediction < 0.5, 0, 255)
    # prediction = prediction.reshape(small_img.shape[0], small_img.shape[1])
    prediction = prediction.astype(np.uint8)
    return prediction
