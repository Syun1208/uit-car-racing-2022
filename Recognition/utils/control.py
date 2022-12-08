import numpy as np
import cv2
import time
import math

# from adafruit_motor import motor, servo
# import digitalio

CHECKPOINT = 45            # Tọa độ hàng lấy ra để tính góc
IMAGESHAPE = [160, 80]      # Kích thước ảnh Input model 
LANEWIGHT = 55            # Độ rộng đường (pixel)




#   Hàm tính góc lại dựa vào ảnh trả về từ model (dự đoán làn đường - không gian màu binary)
def AngleCal(image):
	#   Ý tưởng giải thuật: 
	#       - Trong ảnh binary sẽ lấy 1 dòng (mảng) trong bức ảnh để tìm điểm center của đường (vd: [0 0 0 0 0 0 255 255 255 255 255 255 255 0 0 0 0 0 0 0 0] : 255 - phần đường, 0 - background)
	#       - Đếm xem trong mảng này có bao nhiêu giá trị 255, cộng dồn index của các phần tử 255 lại.
	#       - Lấy cộng dồn index / số lượng giá trị 255 = giá trị tọa độ trung bình của làn đường trong mảng => center
	#       - Có tọa độ Center ta tính góc lệch giữa trung tâm bức ảnh và điểm center => góc lái
	_lineRow = image[CHECKPOINT, :] 
	count = 0
	sumCenter = 0
	centerArg = int(IMAGESHAPE[0]/2)
	minx=0
	maxx=0
	first_flag=True
	for x, y in enumerate(_lineRow):
		if y == 255 and first_flag:
			first_flag=False
			minx=x
		elif y == 255:
			maxx=x
	
	# centerArg = int(sumCenter/count)
	centerArg=int((minx+maxx)//2)
	count=maxx-minx

	# print(minx,maxx,centerArg)
	# print(centerArg, count)

	if (count < LANEWIGHT):
	    if (centerArg < int(IMAGESHAPE[0]/2)):
	        centerArg -= LANEWIGHT - count
	    else :
	        centerArg += LANEWIGHT - count

	image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	_steering = math.degrees(math.atan((centerArg - int(IMAGESHAPE[0]/2))/(IMAGESHAPE[1]-CHECKPOINT)))
	# print(_steering,"----------",count)
	image=cv2.line(image,(centerArg,CHECKPOINT),(int(IMAGESHAPE[0]/2),IMAGESHAPE[1]),(255,0,0),1)
	return image, _steering, centerArg


#   Hàm dự đoán làn đường dựa vào ảnh từ camera và trả về góc lái, center
def road_lines(image, session,inputname):
	# Crop ảnh lại, lấy phần ảnh có làn đường
	image=image[200:,:,:]
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
	copyImage, steering, center = AngleCal(prediction)

	return copyImage, steering, center