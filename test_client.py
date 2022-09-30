from unity_utils.unity_utils import Unity
import cv2
from sympy import limit, Symbol
import time
# import imutils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

unity_api = Unity(11000)
unity_api.connect()
list_area = []
error_arr = np.zeros(5)
list_image = np.zeros(5)
list_angle = np.zeros(5)
t = time.time()


def findingLane(mask):
    arr_normal = []
    height = mask.shape[0] - 50
    lineRow = mask[height, :]
    for x, y in enumerate(lineRow):
        if y == 255:
            arr_normal.append(x)
    minLane = min(arr_normal)
    maxLane = max(arr_normal)
    center = int((minLane + maxLane) / 2)
    error = int(mask.shape[1] / 2) - center
    return error, minLane, maxLane


def bird_view(image):
    width, height = 150, 600
    pts1 = np.float32([[0, 100], [300, 100], [0, 200], [300, 200]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    birdview = cv2.warpPerspective(image, matrix, (height, width))
    return birdview


def computeArea(mask):
    gray = cv2.GaussianBlur(mask, (7, 7), 0)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size_elements = 0
    for cnt in cnts:
        cv2.drawContours(mask, cnts, -1, (0, 0, 255), 3)
        size_elements += cv2.contourArea(cnt)
        list_area.append(cv2.contourArea(cnt))
    print("Area: ", max(list_area))
    return max(list_area)


def ROIStraight(mask):
    height = mask.shape[0]
    width = mask.shape[1]
    polygon = np.array([
        [(420, 0), (0, 149), (0, 0)]
    ])
    polygon2 = np.array([
        [(150, 0), (599, 149), (599, 0)]
    ])
    cv2.fillPoly(mask, polygon, 0)
    cv2.fillPoly(mask, polygon2, 0)
    return mask


def control(image):
    area = computeArea(image)
    if area > 65506.5:
        image = ROIStraight(image)
    error, minLane, maxLane = findingLane(image)
    '''Keep going straight ahead'''
    # if maxLane > 440:
    #     speed = 20
    #     center = int((minLane + maxLane) * 1 / 3)
    # elif minLane < 150:
    #     speed = 20
    #     center = int((minLane + maxLane) * 3 / 4)
    # else:
    #     speed = 20
    #     center = int((minLane + maxLane) / 2)
    # error = int(image.shape[1] / 2) - center
    angle = PID(error)
    # if -1 <= angle <= 1:
    #     speed = 30
    # elif -7 < angle < -1 or 1 < angle < 7:
    #     speed = 10
    # else:
    #     speed = 2
    # print(error)
    list_angle[1:] = list_angle[0:-1]
    list_angle[0] = abs(error)
    list_angle_train = np.array(list_angle).reshape((-1, 1))
    speed = np.dot(list_angle, - 0.1) + 20
    reg = LinearRegression().fit(list_angle_train, speed)
    # reg = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=0)
    speed = reg.predict(np.array(list_angle_train))

    return angle, speed[0]


def is_contour_bad(c):
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # the contour is 'bad' if it is not a rectangle
    return not len(approx) == 0


# def removeNoise(image):
#     edged = cv2.Canny(image, 50, 100)
#     cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     mask = np.ones(image.shape[:2], dtype="uint8") * 255
#     no = cv2.bitwise_not(image)
#     # loop over the contours
#     for c in cnts:
#         cv2.drawContours(mask, [c], -1, 0, -1)
#     image = cv2.bitwise_or(no, no, mask=mask)
#     image = cv2.bitwise_not(image)
#     return image


def PID(error, p=0.15, i=0, d=0.01):
    global t
    global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error * p
    delta_t = time.time() - t
    t = time.time()
    D = (error - error_arr[1]) / delta_t * d
    I = np.sum(error_arr) * delta_t * i
    angle = P + I + D
    if abs(angle) > 5:
        angle = np.sign(angle) * 40
    return - int(angle) * 24 / 50


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


def getDynamicallyAverageImage(image):
    global list_image
    list_image[1:] = list_image[0:-1]
    list_image[0] = image
    avg_image = list_image[0]
    for i in range(len(list_image)):
        if i == 0:
            pass
        else:
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(list_image[i], alpha, avg_image, beta, 0.0)
    return avg_image


def main():
    while True:
        start_time = time.time()
        '''-----------------------Image Processing----------------------------'''
        left_image, right_image = unity_api.get_images()
        kernel = np.ones((15, 15), np.uint8)
        left_image, right_image = convertGreen2White(left_image, right_image)
        # left_image = removeNoise(left_image)
        # right_image = removeNoise(right_image)
        left_image = cv2.dilate(left_image, kernel, iterations=1)
        right_image = cv2.dilate(right_image, kernel, iterations=1)
        # left_image = cv2.threshold(left_image, 45, 255, cv2.THRESH_BINARY)[1]
        # right_image = cv2.threshold(right_image, 45, 255, cv2.THRESH_BINARY)[1]
        # left_image = cv2.erode(left_image, None, iterations=5)
        # right_image = cv2.erode(right_image, None, iterations=5)
        left_image = removeSmallContours(left_image)
        right_image = removeSmallContours(right_image)
        print("time: ", 1 / (time.time() - start_time))
        unity_api.show_images(left_image, right_image)
        image = np.concatenate((left_image, right_image), axis=1)
        cv2.imshow('Predicted Image', image)
        '''--------------------------Controller--------------------------------'''
        angle, speed = control(image)
        computeArea(image)
        image = ROIStraight(image)
        # plt.imshow(image)
        # plt.show()
        cv2.imshow('Predicted Image', image)
        data = unity_api.set_speed_angle(speed, angle)  # speed: [0:100], angle: [-25:25]
        print(data)


if __name__ == "__main__":
    main()
