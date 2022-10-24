import numpy as np
import cv2
from pathlib import Path
import os
import logging
import sys
from itertools import chain
import time
import itertools

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)

list_area = list()
trafficSignsRegister = list()


class imageProcessing:
    def __init__(self, mask, left_mask, right_mask, trafficSigns):
        self.mask = mask
        self.left_mask = left_mask[int(self.mask.shape[0] * 50 / 100):, :]
        self.right_mask = right_mask[int(self.mask.shape[0] * 50 / 100):, :]
        self.trafficSigns = trafficSigns
        self.scale = 0
        self.height = self.mask.shape[0]
        self.width = self.mask.shape[1]

    def __ROIStraight(self):
        polygonLeft = np.array([
            [(600, 0), (0, 150), (0, 0)]
        ])
        polygonRight = np.array([
            [(0, 0), (600, 150), (600, 0)]
        ])
        cv2.fillPoly(self.mask, polygonRight, 0)
        cv2.fillPoly(self.mask, polygonLeft, 0)
        return self.mask

    def __ROITurnLeft(self):
        polygonRight = np.array([
            [(0, 0), (0, 10), (600, 150), (600, 0)]
        ])
        polygonUpper = np.array([
            [(0, 0), (self.width, 0), (self.width, self.height * 1 // 3), (0, self.height * 1 // 3)]
        ])
        cv2.fillPoly(self.mask, polygonUpper, 0)
        cv2.fillPoly(self.mask, polygonRight, 0)
        return self.mask

    def __ROITurnRight(self):
        polygonLeft = np.array([
            [(600, 0), (600, 10), (0, 150), (0, 0)]
        ])
        polygonUpper = np.array([
            [(0, 0), (self.width, 0), (self.width, self.height * 1 // 3), (0, self.height * 1 // 3)]
        ])
        cv2.fillPoly(self.mask, polygonUpper, 0)
        cv2.fillPoly(self.mask, polygonLeft, 0)
        return self.mask

    def __ROINoTurnRight(self):
        polygonRight = np.array([
            [(0, 0), (600, 150), (600, 0)]
        ])
        cv2.fillPoly(self.mask, polygonRight, 0)
        return self.mask

    def __ROINoTurnLeft(self):
        polygonLeft = np.array([
            [(600, 0), (0, 150), (0, 0)]
        ])
        cv2.fillPoly(self.mask, polygonLeft, 0)
        return self.mask

    def __computeArea(self):
        gray = cv2.GaussianBlur(self.mask, (7, 7), 0)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        size_elements = 0
        for cnt in cnts:
            cv2.drawContours(self.mask, cnts, -1, (0, 0, 255), 3)
            size_elements += cv2.contourArea(cnt)
            list_area.append(cv2.contourArea(cnt))
        # print("Area: ", max(list_area))
        return max(list_area)

    def __removeSmallContours(self):
        image_binary = np.zeros((self.mask.shape[0], self.mask.shape[1]), np.uint8)
        contours = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        masked = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        image_remove = cv2.bitwise_and(self.mask, self.mask, mask=masked)
        return image_remove

    def __convertGreen2White(self):
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        bg = cv2.morphologyEx(self.mask, cv2.MORPH_DILATE, se)
        out_gray = cv2.divide(self.mask, bg, scale=255)
        self.mask = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
        self.mask = self.__removeSmallContours()
        return self.mask

    @staticmethod
    def __extend_line(p1, p2, distance=500):
        diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
        p3_x = int(p1[0] + distance * np.cos(diff))
        p3_y = int(p1[1] + distance * np.sin(diff))
        p4_x = int(p1[0] - distance * np.cos(diff))
        p4_y = int(p1[1] - distance * np.sin(diff))
        return [(p3_x, p3_y), (p4_x, p4_y)]

    # rgb(0,255,178)
    @staticmethod
    def calc_x(coeff, y):
        return (y - coeff[1]) / coeff[0]

    @staticmethod
    def get_slope(mask):
        initial_mask = mask
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 120, 240, apertureSize=3)
        minLineLength = 50
        maxLineGap = 20
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength, maxLineGap)
        right_slope = 0
        left_slope = 0
        if lines is not None:
            for line in lines:
                # x1 = line[0, 0]
                # y1 = line[0, 1]
                # x2 = line[0, 2]
                # y2 = line[0, 3]
                # p1, p2 = (line[0, 0], line[0, 1]), (line[0, 2], line[0, 3])
                # (line[0, 0], line[0, 1]), (line[0, 2], line[0, 3]) = self.__extend_line(p1, p2)
                diff_x = (line[0, 0] - line[0, 2])
                if diff_x == 0:
                    continue
                else:
                    slope = (line[0, 1] - line[0, 3]) / diff_x
                    if -0.15 > slope > -0.8:
                        right_slope = slope
                    elif 0.15 < slope < 0.8:
                        left_slope = slope
        return right_slope, left_slope

    def draw_lines(self, mask, lines, color=(0, 0, 255), thickness=3):
        right_lane_lines = []
        left_lane_lines = []
        horizontal_lane_lines = []

        if lines is not None:
            for line in lines:
                # x1 = line[0, 0]
                # y1 = line[0, 1]
                # x2 = line[0, 2]
                # y2 = line[0, 3]
                # p1, p2 = (line[0, 0], line[0, 1]), (line[0, 2], line[0, 3])
                # (line[0, 0], line[0, 1]), (line[0, 2], line[0, 3]) = self.__extend_line(p1, p2)
                diff_x = (line[0, 0] - line[0, 2])
                if diff_x == 0:
                    continue
                else:
                    slope = (line[0, 1] - line[0, 3]) / diff_x
                    if abs(slope) < 0.1:
                        horizontal_lane_lines.append(line)
                        continue
                    elif -0.25 > slope > -0.35:
                        right_lane_lines.append(line)
                    elif 0.15 < slope < 0.35 or 0.65 < slope < 0.75:
                        left_lane_lines.append(line)

        right_lane_lines = np.array(right_lane_lines)
        left_lane_lines = np.array(left_lane_lines)
        horizontal_lane_lines = np.array(horizontal_lane_lines)

        right_check = right_lane_lines.any()
        left_check = left_lane_lines.any()
        horizontal_check = horizontal_lane_lines.any()

        # if horizontal_check:
        #     horizontal_x = list(chain(*horizontal_lane_lines[:,:,0])) + list(chain(*horizontal_lane_lines[:,:,2]))
        #     horizontal_y = list(chain(*horizontal_lane_lines[:,:,1])) + list(chain(*horizontal_lane_lines[:,:,3]))

        #     horizontal_lane_line_coeff = np.polyfit(horizontal_x, horizontal_y, 1)

        #     p5, p6 = (int(self.calc_x(horizontal_lane_line_coeff, mask.shape[0])), mask.shape[0]), \
        #             (int(self.calc_x(horizontal_lane_line_coeff, 350)), 350)
        #     p5, p6 = self.__extend_line(p5, p6)

        #     cv2.line(mask, p5, p6, color, thickness)

        if right_check:
            right_x = list(chain(*right_lane_lines[:, :, 0])) + list(chain(*right_lane_lines[:, :, 2]))
            right_y = list(chain(*right_lane_lines[:, :, 1])) + list(chain(*right_lane_lines[:, :, 3]))

            right_lane_line_coeff = np.polyfit(right_x, right_y, 1)

            p1, p2 = (int(self.calc_x(right_lane_line_coeff, mask.shape[0])), mask.shape[0]), \
                     (int(self.calc_x(right_lane_line_coeff, 350)), 350)
            p1, p2 = self.__extend_line(p1, p2)

            # print(p1, p2)

            cv2.line(mask, p1, p2, color, thickness)

        elif left_check:
            left_x = list(chain(*left_lane_lines[:, :, 0])) + list(chain(*left_lane_lines[:, :, 2]))
            left_y = list(chain(*left_lane_lines[:, :, 1])) + list(chain(*left_lane_lines[:, :, 3]))

            left_lane_line_coeff = np.polyfit(left_x, left_y, 1)

            p3, p4 = (int(self.calc_x(left_lane_line_coeff, mask.shape[0])), mask.shape[0]), \
                     (int(self.calc_x(left_lane_line_coeff, 350)), 350)
            p3, p4 = self.__extend_line(p3, p4)

            # print(p3, p4)

            cv2.line(mask, p3, p4, color, thickness)

        return right_check, left_check, horizontal_check

    def houghLines(self, mask):
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        initial_mask = mask
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 120, 240, apertureSize=3)
        minLineLength = 100
        maxLineGap = 30
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength, maxLineGap)
        mask_lines = np.copy(mask) * 0
        try:
            # for x1,y1,x2,y2 in lines[0]:
            #     print("Coor :", lines[0])
            #     p1, p2 = (x1, y1), (x2, y2)
            #     (x1, y1), (x2, y2) = self.__extend_line(p1, p2)
            #     cv2.line(mask_lines, (x1, y1), (x2, y2), (0,0,255),3)
            #     initial_mask = cv2.addWeighted(initial_mask, 0.8, mask_lines, 1., 0.)
            #     slope = (y2 - y1) / (x2 - x1)
            # return initial_mask, slope
            right_check, left_check, horizontal_check = self.draw_lines(initial_mask, lines)
            return initial_mask, right_check, left_check, horizontal_check
        except Exception as e:
            logging.error(e)
            # return mask_lines

    def __call__(self, *args, **kwargs):
        self.mask = self.__convertGreen2White()
        _, __, __, horizontal_check_left = self.houghLines(self.left_mask)
        _, __, __, horizontal_check_right = self.houghLines(self.right_mask)
        # cv2.imshow('Left', self.left_mask)
        # cv2.imshow('Right', self.right_mask)
        # print('Left: ', horizontal_check_left)
        # print('Right: ', horizontal_check_right)
        # print(minLane, maxLane)
        area = self.__computeArea()
        trafficSignsRegister.insert(0, self.trafficSigns)
        if area >= 67000:
            # if self.trafficSigns == 'turn_right' or self.trafficSigns == 'no_left':
            #     self.mask = self.__ROITurnRight()
            #     delaySign = self.trafficSigns
            #     delayTime = time.time()
            # elif self.trafficSigns == 'turn_left' or self.trafficSigns == 'no_right':
            #     self.mask = self.__ROITurnLeft()
            #     delaySign = self.trafficSigns
            #     delayTime = time.time()
            # else:
            #     self.mask = self.__ROIStraight()
            #     if delaySign == 'turn_right' or self.trafficSigns == 'no_left':
            #         self.mask = self.__ROITurnRight()
            #     elif delaySign == 'turn_left' or self.trafficSigns == 'no_right':
            #         self.mask = self.__ROITurnLeft()
            #     if time.time() - delayTime > 10:
            #         delaySign = None
            # print('Delay sign: ', delaySign)
            # if self.trafficSigns:
            #     if self.trafficSigns == 'turn_right' or self.trafficSigns == 'no_left':
            #         self.mask = self.__ROITurnRight()
            #         trafficSignsRegister.insert(0, self.trafficSigns)
            #     elif self.trafficSigns == 'turn_left' or self.trafficSigns == 'no_right':
            #         self.mask = self.__ROITurnLeft()
            #         trafficSignsRegister.insert(0, self.trafficSigns)
            #     else:
            #         self.mask = self.__ROIStraight()
            #         trafficSignsRegister.insert(0, self.trafficSigns)
            # else:
            #     print(trafficSignsRegister)
            #     print('turn right: ', bool('turn_right' or 'no_left' in trafficSignsRegister))
            #     print('turn left: ', bool('turn_left' or 'no_right' in trafficSignsRegister))
            #     if 'turn_right' or 'no_left' in trafficSignsRegister:
            #         self.mask = self.__ROITurnRight()
            #     elif 'turn_left' or 'no_right' in trafficSignsRegister:
            #         self.mask = self.__ROITurnLeft()
            #     else:
            #         self.mask = self.__ROIStraight()
            #     if len(trafficSignsRegister) > 100:
            #         trafficSignsRegister.pop(-1)
            if self.trafficSigns == 'turn_left' or 'turn_left' in trafficSignsRegister:
                # print('Turn left')
                self.mask = self.__ROITurnLeft()
                self.scale = 37
            elif self.trafficSigns == 'turn_right' or 'turn_right' in trafficSignsRegister:
                # print('Turn right')
                self.mask = self.__ROITurnRight()
                self.scale = 37
            # elif self.trafficSigns == 'straight' or 'straight' in trafficSignsRegister:
            #     self.mask = self.__ROIStraight()
            #     self.scale = 21
            elif self.trafficSigns == 'no_straight':
                # if minLane < 20 and maxLane <= 420:
                #     self.mask = self.__ROITurnLeft()
                #     trafficSignsRegister.insert(0, 'turn_left')
                #     self.scale = 70
                # if maxLane > 300 and minLane >= 10:
                if horizontal_check_right:
                    self.mask = self.__ROITurnRight()
                    trafficSignsRegister.insert(0, 'turn_right')
                elif horizontal_check_left:
                    self.mask = self.__ROITurnLeft()
                    trafficSignsRegister.insert(0, 'turn_left')
                self.scale = 37
            elif self.trafficSigns == 'no_right' or 'no_right' in trafficSignsRegister:
                self.mask = self.__ROINoTurnRight()
                self.scale = 37
            elif self.trafficSigns == 'no_left' or 'no_left' in trafficSignsRegister:
                self.mask = self.__ROINoTurnLeft()
                self.scale = 37
            else:
                self.mask = self.__ROIStraight()
                self.scale = 27.6
        if len(trafficSignsRegister) > 25:
            trafficSignsRegister.pop(-1)
        kernel = np.ones((15, 15), np.uint8)
        self.mask = cv2.dilate(self.mask, kernel, iterations=1)
        self.mask = self.__removeSmallContours()
        return self.mask, self.scale


if __name__ == "__main__":
    ls = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'no_straight',
          'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight',
          'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight',
          'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight',
          'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight',
          'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight',
          'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight', 'no_straight',
          'no_straight', 'no_straight', 'no_straight', 'no_straight']
    ls.insert(0, 'turn_left')
    print(ls)
