import sys

sys.path.insert(0, '/home/long/Desktop/ITCar-PHOLOTINO')

import time
import numpy as np
import cv2
from sklearn.ensemble import RandomForestRegressor
from utils.image_processing import imageProcessing
from skfuzzy import control
from fuzzy_control.fuzzy import Fuzzy
import skfuzzy as fuzzy

t = time.time()
error_arr = np.zeros(5)
list_angle = np.zeros(5)


class Controller(imageProcessing, Fuzzy):
    def __init__(self, mask):
        super(Controller, self).__init__(mask=mask)
        Fuzzy.__init__(self)
        self.mask = self.mainImageProcessing()

    def __findingLane(self):
        arr_normal = []
        height = self.mask.shape[0] - 70
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
        return error

    def __PID(self, p=0.25, i=0, d=0.01):
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
        angle = self.__optimizeFuzzy(angle)
        # if abs(angle) > 5:
        #     angle = np.sign(angle) * 40
        return - int(angle) * 29 / 50

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
        angle = 3.5 * angle
        if abs(angle) > 5:
            angle = np.sign(angle) * 40
        return angle

    '''Angle and Speed Processing'''

    @staticmethod
    def __conditionalSpeed(angle, error):
        list_angle[1:] = list_angle[0:-1]
        list_angle[0] = abs(error)
        list_angle_train = np.array(list_angle).reshape((-1, 1))
        speed = np.dot(list_angle, - 0.1) + 50
        # reg = LinearRegression().fit(list_angle_train, speed)
        reg = RandomForestRegressor(n_estimators=30, random_state=1).fit(list_angle_train, speed)
        predSpeed = reg.predict(np.array(list_angle_train))
        if angle <= -7 or angle >= 7:
            predSpeed[0] = 1
            return angle, predSpeed[0]
        return angle, predSpeed[0]

    def __call__(self, *args, **kwargs):
        # cv2.imshow('Predicted Image', self.mask)
        error = self.__findingLane()
        angle = self.__PID()
        angle, speed = self.__conditionalSpeed(angle, error)
        print("Speed RF: ", speed)
        # predSpeed[0], angle = self.__fuzzy(predSpeed[0], angle)
        return angle, speed
