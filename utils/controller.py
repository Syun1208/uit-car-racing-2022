import sys
import time
import numpy as np
import cv2
from sklearn.ensemble import RandomForestRegressor
from utils.image_processing import imageProcessing
from skfuzzy import control
from fuzzy_control.fuzzy import Fuzzy
import skfuzzy as fuzzy
import os
from pathlib import Path
from utils.traffic_signs_recognition import trafficSignsRecognition

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

t = time.time()
pre_time = time.time()
error_arr = np.zeros(5)
list_angle = np.zeros(5)
width = np.zeros(10)


class Controller(imageProcessing, Fuzzy):
    def __init__(self, mask, trafficSigns):
        imageProcessing.__init__(self, mask, trafficSigns)
        Fuzzy.__init__(self)
        self.mask = imageProcessing.__call__(self)
        self.trafficSigns = trafficSigns

    def findingLane(self, scale=45):
        arr_normal = []
        height = self.mask.shape[0] - scale
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr_normal.append(x)
        if not arr_normal:
            arr_normal = [self.mask.shape[1] * 1 / 3, self.mask.shape[1] * 2 / 3]
        minLane = min(arr_normal)
        maxLane = max(arr_normal)
        center = int((minLane + maxLane) / 2)
        error = int(self.mask.shape[1] / 2) - center
        return error, minLane, maxLane

    def computeError(self, center):
        return int(self.mask.shape[1] / 2) - center

    @staticmethod
    def PID(error, scale=26, p=0.2, i=0, d=0.01):
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
        # angle = self.__optimizeFuzzy(angle)
        if abs(angle) > 5:
            angle = np.sign(angle) * 40
        return - int(angle) * scale / 50

    @staticmethod
    def __fuzzy(speed_car, angle_car):
        speed = control.Antecedent(np.arange(0, 101, 1), 'speed')
        angle = control.Antecedent(np.arange(-25, 26), 'angle')
        tip = control.Consequent(np.arange(0, 26, 1), 'tip')
        speed.automf(3)
        angle.automf(5)
        tip['low'] = fuzzy.trimf(tip.universe, [0, 0, 10])
        tip['medium'] = fuzzy.trimf(tip.universe, [0, 10, 27])
        tip['high'] = fuzzy.trimf(tip.universe, [10, 27, 27])
        rule1 = control.Rule(speed['good'] & angle['poor'] & angle['good'], tip['low'])
        rule2 = control.Rule(speed['poor'] & angle['average'], tip['medium'])
        rule3 = control.Rule(speed['good'] & angle['average'] | speed['poor'] & angle['good'] | angle['poor'],
                             tip['high'])
        tipping_ctrl = control.ControlSystem([rule1, rule2, rule3])
        tipping = control.ControlSystemSimulation(tipping_ctrl)
        tipping.input['speed'] = speed_car
        tipping.input['angle'] = angle_car
        tipping.compute()
        speed_car = speed_car * tipping.output['tip']
        angle_car = angle_car * tipping.output['tip']
        print('Tipping: ', tipping.output['tip'])
        return speed_car, angle_car

    def __optimizeFuzzy(self, angle):
        angle = self.run_fuzzy_controller(angle)
        angle = 3.2 * angle
        if abs(angle) > 5:
            angle = np.sign(angle) * 40
        return angle

    '''Angle and Speed Processing'''

    @staticmethod
    def __conditionalSpeed(angle, error):
        list_angle[1:] = list_angle[0:-1]
        list_angle[0] = abs(error)
        list_angle_train = np.array(list_angle).reshape((-1, 1))
        predSpeed = np.dot(list_angle, - 0.1) + 30
        # reg = LinearRegression().fit(list_angle_train, speed)
        reg = RandomForestRegressor(n_estimators=40, random_state=0).fit(list_angle_train, predSpeed)
        predSpeed = reg.predict(np.array(list_angle_train))

        return angle, predSpeed[0]

    def __call__(self, scale=28, *args, **kwargs):
        # cv2.imshow('Predicted Image', self.mask)
        error, minLane, maxLane = self.findingLane()
        if self.trafficSigns == '' or self.trafficSigns != 'straight':
            print('Lmao')
            angle = self.PID(error, scale=40)
        else:
            angle = self.PID(error, scale=scale)
        angle, speed = self.__conditionalSpeed(angle, error)
        print("Speed RF: ", speed)
        return angle, speed


class TrafficSignsController(Controller):
    def __init__(self, mask, trafficSigns, speed):
        super(TrafficSignsController, self).__init__(mask, trafficSigns)
        self.trafficSigns = trafficSigns
        self.speed = speed
        self.angle = 0
        self.corner = 0
        self.UN_MIN_1 = 10
        self.OV_MIN_1 = 30
        self.UN_MAX_1 = 40
        self.OV_MAX_1 = 50
        self.check = 0
        self.center = 0
        self.MAX_SPEED = 30
        self.width_road = 70
        self.count = 0
        self.centerLeft = 0
        self.centerRight = 600
        self.error = 25
        self.errorLane, self.minLane, self.maxLane = self.findingLane()
        self.errorHead, self.minHead, self.maxHead = self.findingLane(80)

    def __straight(self, underSendBack, optionSpeed):
        if 320 <= self.maxLane <= 500 and 2 <= self.minLane <= 290 and not self.error and not self.corner:
            width[1:] = width[0:-1]
            if self.maxLane - self.maxLane > 60:
                width[0] = self.maxLane - self.minLane
        self.width_road = np.average(width)
        print('Width road: ', self.width_road)
        self.center = int((self.minLane + self.maxLane) / 2)
        if not self.minHead == self.maxHead == 91:
            if self.maxLane >= self.OV_MAX_1 and self.UN_MIN_1 <= self.minLane <= self.OV_MIN_1:
                self.center = self.minLane + int(self.width_road / 2)
            elif self.minLane < self.UN_MIN_1 and self.UN_MAX_1 <= self.maxLane <= self.OV_MAX_1:
                self.center = self.maxLane - int(self.width_road / 2)
        self.speed = self.__PWM()
        if float(self.speed) < 20.0:
            self.speed = 30
        elif float(self.speed) > optionSpeed:  # Adjust Speed
            self.speed = underSendBack
        if self.minLane == 450 and self.maxLane == 150 or self.maxLane == 550 or self.minLane == 0:
            self.count += 1
        return self.speed, self.center

    def __turnRight(self):
        if not self.corner and self.count:
            self.corner = 1

        if self.corner:
            if time.time() - pre_time < 1.0:
                self.center = self.centerRight
            else:
                self.trafficSigns = 'straight'
                self.__reset()

        return self.trafficSigns, self.center

    def __turnLeft(self):
        if not self.corner and self.count:
            self.corner = 1

        if self.corner:
            if time.time() - pre_time < 1.0:
                self.center = self.centerLeft
            else:
                self.trafficSigns = 'straight'
                self.__reset()

        return self.trafficSigns, self.center

    def __PWM(self):
        return -3 * abs(self.error) + 20

    def __maxSpeedFunction(self):
        return -0.125 * abs(self.error) + 20

    def __controlTurning(self):
        pass

    def __reset(self):
        self.corner = 0
        self.check = 0
        self.count = 0

    def __call__(self, *args, **kwargs):
        # self.MAX_SPEED = self.__maxSpeedFunction()
        # if self.trafficSigns == 'decrease':
        #     self.speed, self.center = self.__straight(-10, 20)
        #     self.__reset()
        # elif self.trafficSigns == 'straight':
        #     self.speed, self.center = self.__straight(10, self.MAX_SPEED)
        #     self.__reset()
        # elif self.trafficSigns == 'no_straight':
        #     self.speed, self.center = self.__straight(0, 10)
        #     if not self.check:
        #         if self.minLane <= 10:
        #             self.check = 1
        #         elif self.maxLane >= 600:
        #             self.check = 2
        #     elif self.check == 2:
        #         self.trafficSigns, self.center = self.__turnRight()
        #     else:
        #         self.trafficSigns, self.center = self.__turnLeft()
        # elif self.trafficSigns == 'turn_right' or self.trafficSigns == 'no_turn_left':
        #     self.speed, self.center = self.__straight(0, 10)
        #     if self.maxLane >= 134 and not self.check:
        #         self.check = 1
        #     elif self.check:
        #         self.trafficSigns, self.center = self.__turnRight()
        # elif self.trafficSigns == 'turn_left' or self.trafficSigns == 'no_turn_right':
        #     self.trafficSigns, self.center = self.__straight(0, 10)
        #     if self.minLane <= 25 and not self.check:
        #         self.check = 1
        #     elif self.check:
        #         self.trafficSigns, self.center = self.__turnLeft()
        # elif self.trafficSigns == 'car_right':
        #     self.trafficSigns, self.center = self.__straight(10, self.MAX_SPEED)
        #     self.center -= 5
        # elif self.trafficSigns == 'car_left':
        #     self.trafficSigns, self.center = self.__straight(10, self.MAX_SPEED)
        #     self.center += 5
        # if self.trafficSigns == 'straight':
        #     print('True')
        #     self.angle, self.speed = Controller.__call__(self)
        # elif self.trafficSigns == 'turn_right':
        #     print('Haha')
        #     self.center = 600
        #     self.error = self.computeError(self.center)
        #     self.angle, self.speed = Controller.__call__(self)
        # elif self.trafficSigns == 'turn_left':
        #     print('Haha')
        #     self.center = 0
        #     self.error = self.computeError(self.center)
        #     self.angle = self.PID(self.error, scale=40)
        #     self.angle, self.speed = Controller.__call__(self)
        # self.error = self.computeError(self.center)
        # angle = self.PID(self.error)
        return self.trafficSigns, self.speed, self.angle
