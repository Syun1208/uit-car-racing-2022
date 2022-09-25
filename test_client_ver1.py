from gc import collect
from itertools import count
from unity_utils.unity_utils import Unity
import cv2
import time
import imutils
from cmath import atan
import math
from collections import Counter
import collections
import numpy as np

unity_api = Unity(11000)
unity_api.connect()


def contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=10)
    thresh = cv2.dilate(thresh, None, iterations=3)
    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    try:
        c = max(cnts, key=cv2.contourArea)
    except ValueError:
        return 0, 0, 0, 0, 0
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][-1])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    extMid = []
    extMid.append(int((extRight[0] + extBot[0]) / 2))
    extMid.append(int((extRight[1] + extBot[1]) / 2))
    extMid = tuple(extMid)

    cv2.drawContours(image, [c], -1, (0, 255, 100), 2)
    cv2.circle(image, extLeft, 4, (0, 0, 255), -1)
    cv2.circle(image, extRight, 8, (0, 55, 10), -1)
    cv2.circle(image, extTop, 4, (255, 100, 0), -1)
    cv2.circle(image, extBot, 4, (100, 255, 0), -1)
    cv2.circle(image, extMid, 20, (200, 27, 50), -1)
    # show the output image
    distopleft = math.sqrt((extLeft[0] - extTop[0]) ** 2 + (extLeft[1] - extTop[1]) ** 2)
    distopbot = (extMid[1] - extTop[1])
    angmid = (-atan((extMid[0] - extTop[0]) / (extMid[1] - extTop[1])) * 180 / math.pi) ** 0.7722
    a = np.all(image[148, :, :] == [0, 0, 0], axis=1)
    countimg = Counter(a)[0]
    print('anh xanh ', countimg)
    # countimg=(np.count_nonzero(image[:][148]==(0,0,0)))
    # print(countimg)
    print('khoang cach ', distopbot, distopleft)
    # if(collections.Counter(image[130][4]== 0,0,0)):
    #     print('hihi')

    # cv2.imshow("Image", image)
    return angmid, image, countimg, distopbot, distopleft


def mixspeed(ang):
    if ang > 20:
        speed = 5
    elif (ang > 8 and ang <= 20):
        speed = 10
    else:
        speed = 17
    return speed


angleft, angright = 0, 0
while True:

    start_time = time.time()
    left_image, right_image = unity_api.get_images()
    # cv2.imshow('ff',image)
    # cv2.imwrite('ga.jpg',left_image)
    img_full = cv2.resize(left_image, dsize=(600, 150))
    print('shape', img_full.shape)
    # img_full[:,:300,:]=left_image
    # img_full[:,300:600,:]=right_image[:,:,:]
    angleft, left_image, acountimg, adistopbot, adistopleft = contour(left_image)
    angright, right_image, bcountimg, bdistopbot, bdistopleft = contour(right_image)
    # cv2.imshow('img',img_full)
    # print("time: ", 1/(time.time() - start_time))
    if acountimg == 0:
        data = unity_api.set_speed_angle(1, 25)
        print('turn left 3\n')
        continue
    elif bcountimg == 0:
        data = unity_api.set_speed_angle(1, -25)
        print('turn right 3\n')
        continue
    elif bdistopbot > 55 and adistopbot > 55 and abs(adistopbot - bdistopbot) < 10 and abs(acountimg - bcountimg) < 30:
        data = unity_api.set_speed_angle(20, 0)
        print(' ahead 3\n')
        continue
    # elif  bdistopbot<54 and adistopleft <250 and acountimg>220:
    #     data = unity_api.set_speed_angle(5, -15)
    #     print(' ngatu 3\n')
    #     continue
    elif bdistopleft == 0 and adistopleft > 285 and adistopbot > 65 and acountimg > 295:
        ata = unity_api.set_speed_angle(5, 0)
        print(' right 3\n')
        continue
    elif bcountimg <= 200 and acountimg >= 280:
        data = unity_api.set_speed_angle(3, -23.5)
        print('turn left\n')
        continue
    elif acountimg <= 200 and bcountimg >= 280:
        data = unity_api.set_speed_angle(3, 23.5)
        print('turn right\n')
        continue
    ang = (angleft.real + angright.real) * 1.2
    speed = mixspeed(abs(ang))
    data = unity_api.set_speed_angle(speed, ang)
    print(data, '\n')
    # cv2.imshow('img',img_full)
    unity_api.show_images(left_image, right_image)
