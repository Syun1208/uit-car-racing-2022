from unity_utils.unity_utils import Unity
import cv2
import time
import imutils
from PIL import Image
import numpy as np

unity_api = Unity(11000)
unity_api.connect()


# def PID(error, p=0.35, i=0, d=0.01):
#     self.error_arr[1:] = self.error_arr[0:-1]
#     self.error_arr[0] = error
#     P = error * self.p
#     delta_t = time.time() - self.time
#     D = (error - self.error_arr[1]) / delta_t * self.d
#     I = np.sum(self.error_arr) * delta_t * self.i
#     angle = P + I + D
#     if abs(angle) > 5:
#         angle = np.sign(angle) * 40
#     return int(angle)


def convertGreen2White(left_image, right_image):
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    se_left = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg_left = cv2.morphologyEx(left_image, cv2.MORPH_DILATE, se_left)
    out_gray_left = cv2.divide(left_image, bg_left, scale=255)
    out_binary_left = cv2.threshold(out_gray_left, 0, 255, cv2.THRESH_OTSU)[1]

    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    se_right = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg_right = cv2.morphologyEx(right_image, cv2.MORPH_DILATE, se_right)
    out_gray_right = cv2.divide(right_image, bg_right, scale=255)
    out_binary_right = cv2.threshold(out_gray_right, 0, 255, cv2.THRESH_OTSU)[1]

    return out_binary_left, out_binary_right


def removeSmallContours(mask):
    image_binary = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    masked = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(mask, mask, mask=masked)
    return image_remove


def main():
    while True:
        data = unity_api.set_speed_angle(150, 0)  # speed: [0:150], angle: [-25:25]
        print(data)
        start_time = time.time()
        left_image, right_image = unity_api.get_images()
        kernel = np.ones((15, 15), np.uint8)
        left_image = cv2.dilate(left_image, kernel, iterations=1)
        right_image = cv2.dilate(right_image, kernel, iterations=1)
        left_image, right_image = convertGreen2White(left_image, right_image)
        left_image = removeSmallContours(left_image)
        right_image = removeSmallContours(right_image)
        print("time: ", 1 / (time.time() - start_time))
        unity_api.show_images(left_image, right_image)


if __name__ == "__main__":
    main()
