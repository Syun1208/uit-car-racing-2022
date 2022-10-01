import numpy as np
import cv2

list_area = list()


class imageProcessing:
    def __init__(self, mask):
        self.mask = mask

    def __ROIStraight(self):
        height = self.mask.shape[0]
        width = self.mask.shape[1]
        polygonRight = np.array([
            [(450, 0), (0, 150), (0, 0)]
        ])
        polygonLeft = np.array([
            [(150, 0), (600, 150), (600, 0)]
        ])
        cv2.fillPoly(self.mask, polygonRight, 0)
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

    def mainImageProcessing(self, *args, **kwargs):
        self.__convertGreen2White()
        area = self.__computeArea()
        self.mask = self.__ROIStraight()
        kernel = np.ones((15, 15), np.uint8)
        self.mask = cv2.dilate(self.mask, kernel, iterations=1)
        self.mask = self.__removeSmallContours()
        return self.mask
