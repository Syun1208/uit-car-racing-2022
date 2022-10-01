import time
import numpy as np
import cv2
from sklearn.ensemble import RandomForestRegressor
from utils.image_processing import imageProcessing

t = time.time()
error_arr = np.zeros(5)
list_angle = np.zeros(5)


class Controller(imageProcessing):
    def __init__(self, mask):
        super(Controller, self).__init__(mask=mask)
        self.mask = self.mainImageProcessing()

    def __findingLane(self):
        arr_normal = []
        height = self.mask.shape[0] - 50
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr_normal.append(x)
        minLane = min(arr_normal)
        maxLane = max(arr_normal)
        center = int((minLane + maxLane) / 2)
        error = int(self.mask.shape[1] / 2) - center
        return error

    def __PID(self, p=0.15, i=0, d=0.01):
        global t
        global error_arr
        error = self.__findingLane()
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
        return - int(angle) * 27 / 50

    def __call__(self, *args, **kwargs):
        cv2.imshow('Predicted Image', self.mask)
        error = self.__findingLane()
        angle = self.__PID()
        list_angle[1:] = list_angle[0:-1]
        list_angle[0] = abs(error)
        list_angle_train = np.array(list_angle).reshape((-1, 1))
        speed = np.dot(list_angle, - 0.1) + 23
        # reg = LinearRegression().fit(list_angle_train, speed)
        reg = RandomForestRegressor(n_estimators=30, random_state=0).fit(list_angle_train, speed)
        predSpeed = reg.predict(np.array(list_angle_train))
        return angle, predSpeed[0]
