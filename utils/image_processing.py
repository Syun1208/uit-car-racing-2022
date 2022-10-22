import numpy as np
import cv2
import time
import itertools

list_area = list()
trafficSignsRegister = list()


class imageProcessing:
    def __init__(self, mask, trafficSigns):
        self.mask = mask
        self.trafficSigns = trafficSigns
        self.height = self.mask.shape[0]
        self.width = self.mask.shape[1]

    def __ROIStraight(self):
        polygonLeft = np.array([
            [(500, 0), (0, 150), (0, 0)]
        ])
        polygonRight = np.array([
            [(100, 0), (600, 150), (600, 0)]
        ])
        cv2.fillPoly(self.mask, polygonRight, 0)
        cv2.fillPoly(self.mask, polygonLeft, 0)
        return self.mask

    def __ROITurnLeft(self):
        polygonRight = np.array([
            [(100, 0), (600, 150), (600, 0)]
        ])
        polygonUpper = np.array([
            [(0, 0), (self.width, 0), (self.width, self.height * 2 // 3), (0, self.height * 2 // 3)]
        ])
        cv2.fillPoly(self.mask, polygonUpper, 0)
        cv2.fillPoly(self.mask, polygonRight, 0)
        return self.mask

    def __ROITurnRight(self):
        polygonLeft = np.array([
            [(500, 0), (0, 150), (0, 0)]
        ])
        polygonUpper = np.array([
            [(0, 0), (self.width, 0), (self.width, self.height * 2 // 3), (0, self.height * 2 // 3)]
        ])
        cv2.fillPoly(self.mask, polygonUpper, 0)
        cv2.fillPoly(self.mask, polygonLeft, 0)
        return self.mask

    def __ROINoTurnRight(self):
        polygonRight = np.array([
            [(100, 0), (600, 150), (600, 0)]
        ])
        cv2.fillPoly(self.mask, polygonRight, 0)
        return self.mask

    def __ROINoTurnLeft(self):
        polygonLeft = np.array([
            [(500, 0), (0, 150), (0, 0)]
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
        print("Area: ", max(list_area))
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

    def __call__(self, minLane=0, maxLane=600, *args, **kwargs):
        self.__convertGreen2White()
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
                self.mask = self.__ROITurnLeft()
            elif self.trafficSigns == 'turn_right' or 'turn_right' in trafficSignsRegister:
                self.mask = self.__ROITurnRight()
            elif self.trafficSigns == 'straight' or 'straight' in trafficSignsRegister:
                self.mask = self.__ROIStraight()
            elif self.trafficSigns == 'no_straight' or 'no_straight' in trafficSignsRegister:
                if minLane <= 10 and maxLane <= 600:
                    self.mask = self.__ROITurnLeft()
                elif maxLane >= 600 and minLane >= 10:
                    self.mask = self.__ROITurnRight()
            elif self.trafficSigns == 'no_right' or 'no_right' in trafficSignsRegister:
                self.mask = self.__ROINoTurnRight()
            elif self.trafficSigns == 'no_left' or 'no_left' in trafficSignsRegister:
                self.mask = self.__ROINoTurnLeft()
        if len(trafficSignsRegister) > 90:
            trafficSignsRegister.pop(-1)
        kernel = np.ones((15, 15), np.uint8)
        self.mask = cv2.dilate(self.mask, kernel, iterations=1)
        self.mask = self.__removeSmallContours()
        return self.mask
