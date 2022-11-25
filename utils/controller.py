import sys
import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from utils.image_processing import imageProcessing
from skfuzzy import control
from IPython.display import clear_output
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
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)

t = time.time()
pre_time = time.time()
data = dict()
list_data_angle = list()
list_data_predicted_speed = list()
list_data_expected_speed = list()
list_data_error = list()
error_arr = np.zeros(5)
list_angle = np.zeros(5)
width = np.zeros(10)


class Controller(imageProcessing, trafficSignsRecognition, Fuzzy):
    def __init__(self, mask, current_speed):
        trafficSignsRecognition.__init__(self, mask)
        self.trafficSigns = trafficSignsRecognition.__call__(self)
        imageProcessing.__init__(self, mask, mask[:, :mask.shape[1] // 2], mask[:, mask.shape[1] // 2:],
                                 self.trafficSigns)
        Fuzzy.__init__(self)
        self.mask, self.scale = imageProcessing.__call__(self)
        self.current_speed = current_speed

    def __reduceSpeed(self, speed):
        if self.current_speed > 30:
            return -2
        else:
            return speed

    def findingLane(self, scale=60):
        arr_normal = []
        height = self.mask.shape[0] - scale
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr_normal.append(x)
        if not arr_normal:
            arr_normal = [self.mask.shape[1] * 1 // 3, self.mask.shape[1] * 2 // 3]
        minLane = min(arr_normal)
        maxLane = max(arr_normal)
        center = int((minLane + maxLane) / 2)
        error = int(self.mask.shape[1] / 2) - center
        return error

    def computeError(self, center):
        return int(self.mask.shape[1] / 2) - center

    @staticmethod
    def __PID(error, scale=28, p=0.17, i=0, d=0.01):
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

    def __conditionalSpeed(self, angle, error):
        list_angle[1:] = list_angle[0:-1]
        list_angle[0] = abs(error)
        list_angle_train = np.array(list_angle).reshape((-1, 1))
        predSpeed = np.dot(list_angle, - 0.1) + 28
        # predSpeed = np.add(np.dot(np.power(list_angle, 2), -0.1), np.dot(list_angle, 20)) + 50
        # list_data_expected_speed.append(np.average(predSpeed, axis=0))
        # data['expected_speed'] = list_data_expected_speed
        # reg = LinearRegression().fit(list_angle_train, speed)
        reg = RandomForestRegressor(n_estimators=40, random_state=0).fit(list_angle_train, predSpeed)
        predSpeed = reg.predict(np.array(list_angle_train))
        # list_data_predicted_speed.append(predSpeed[0])
        # list_data_angle.append(angle)
        # list_data_error.append(error)
        # data['predicted_speed'] = list_data_predicted_speed
        # data['angle'] = list_data_angle
        # data['error'] = list_data_error
        # df = pd.DataFrame(data)
        # df.to_csv(os.path.join(str(WORK_DIR), 'data/data_linux.csv'), index=False)
        # clear_output(wait=True)
        # X_grid = np.arange(min(list_angle_train), max(list_angle_train), 0.01)
        # X_grid = X_grid.reshape((len(X_grid), 1))
        # plt.scatter(list_angle_train, predSpeed, color='red')
        # plt.plot(X_grid, reg.predict(X_grid), color='blue')
        # plt.title('Random Forest Regression Model')
        # plt.xlabel('Error')
        # plt.ylabel('Speed')
        # plt.savefig(ROOT / 'random_forest.png')
        if self.trafficSigns != '':
            print('QÚA GHÊ GỚM ! VÀ ĐÂY LÀ PHOLOTINO !')
            if self.trafficSigns != 'straight' or angle < -4 or angle > 4:
                start = time.time()
                predSpeed[0] = 3
                if time.time() - start > 3:
                    predSpeed[0] = 10
            elif self.trafficSigns == 'straight':
                predSpeed[0] = 28
        return predSpeed[0]

    def __call__(self, *args, **kwargs):
        # cv2.imshow('Predicted Image', self.mask)
        error = self.findingLane()
        if self.trafficSigns != '' or self.trafficSigns != 'straight':
            error = self.findingLane(scale=42)
        angle = self.__PID(error, self.scale)
        speed = self.__conditionalSpeed(angle, error)
        # if speed >= 30:
        #     speed = speed - 25
        speed = self.__reduceSpeed(speed)
        # print("Speed RF: ", speed)
        return angle, speed


class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 20
        self.dt = 0.2
        # every element in input array u will be remained for dt seconds
        # here with horizon 20 and dt 0.2 we will predict 4 seconds ahead(20*0.2)
        # we can't predict too much ahead in time because that might be pointless and take too much computational time
        # we can't predict too less ahead in time because that might end up overshooting from end point as it won't be able to see the end goal in time

        # Reference or set point the controller will achieve.
        self.reference = [50, 0, 0]

    @staticmethod
    def plant_model(prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        v_t = prev_state[3]  # m/s
        a_t = pedal

        x_t_1 = x_t + v_t * dt  # distance = speed*time
        v_t_1 = v_t + a_t * dt - v_t / 25.0  # v = u + at (-v_t/25 is a rough estimate for friction)

        return [x_t_1, 0, 0, v_t_1]

    def cost_function(self, u, *args):
        state = args[0]
        ref = args[1]
        x_end = ref[0]
        cost = 0.0

        for i in range(self.horizon):
            state = self.plant_model(state, self.dt, u[i * 2], u[i * 2 + 1])
            # u input vector is designed like = [(pedal for t1), (steering for t1), (pedal for t2), (steering for t2)...... (pedal for t(horizon)), (steering for t(horizon))]
            x_current = state[0]
            v_current = state[3]

            cost += (x_end - x_current) ** 2  # so we keep getting closer to end point

            if v_current * 3.6 > 10:  # speed limit is 10km/h, 3.6 is multiplied to convert m/s to km/h
                cost += 100 * v_current

            if x_current > x_end:  # so we don't overshoot further than end point
                cost += 10000

        return cost

    def __call__(self, *args, **kwargs):
        start = time.process_time()
        num_inputs = 2
        # u: desire output
        # y: predicted output
        u = np.zeros(self.horizon * num_inputs)
        bounds = []
        # Set bounds for inputs bounded optimization.
        for i in range(self.horizon):
            bounds += [[-1, 1]]
            bounds += [[-0.0, 0.0]]
        ref = self.reference
        state_i = np.array([[1, 0, 0, 0]])
        u_i = np.array([[0, 0]])
        sim_total = 100
        predict_info = [state_i]
        for i in range(1, sim_total + 1):
            # Reuse old inputs as starting point to decrease run time.
            u = np.delete(u, 0)
            u = np.delete(u, 0)
            u = np.append(u, u[-2])
            u = np.append(u, u[-2])
            u = np.zeros(self.horizon * num_inputs)
            start_time = time.time()
            # Non-linear optimization.
            u_solution = minimize(self.cost_function, u, (state_i[-1], ref),
                                  method='SLSQP',
                                  bounds=bounds,
                                  tol=1e-5)
            print('Step ' + str(i) + ' of ' + str(sim_total) + '   Time ' + str(round(time.time() - start_time, 5)))
            u = u_solution.x
            y = self.plant_model(state_i[-1], self.dt, u[0], u[1])
            predicted_state = np.array([y])
            for j in range(1, self.horizon):
                predicted = self.plant_model(predicted_state[-1], self.dt, u[2 * j], u[2 * j + 1])
                predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
            # Output
            predict_info += [predicted_state]
            state_i = np.append(state_i, np.array([y]), axis=0)
            u_i = np.append(u_i, np.array([(u[0], u[1])]), axis=0)
        return np.average(predict_info), np.average(u_i)


class DeepQLearning:
    def __init__(self, speed, angle):
        self.speed = speed
        self.angle = angle

    def __call__(self, *args, **kwargs):
        pass


class QLearning:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

# class TrafficSignsController(Controller):
#     def __init__(self, mask, trafficSigns, speed):
#         super(TrafficSignsController, self).__init__(mask, trafficSigns)
#         self.trafficSigns = trafficSigns
#         self.speed = speed
#         self.angle = 0
#         self.corner = 0
#         self.UN_MIN_1 = 10
#         self.OV_MIN_1 = 30
#         self.UN_MAX_1 = 40
#         self.OV_MAX_1 = 50
#         self.check = 0
#         self.center = 0
#         self.MAX_SPEED = 30
#         self.width_road = 70
#         self.count = 0
#         self.centerLeft = 0
#         self.centerRight = 600
#         self.error = 25
#         self.errorLane, self.minLane, self.maxLane = self.findingLane()
#         self.errorHead, self.minHead, self.maxHead = self.findingLane(80)
#
#     def __straight(self, underSendBack, optionSpeed):
#         if 320 <= self.maxLane <= 500 and 2 <= self.minLane <= 290 and not self.error and not self.corner:
#             width[1:] = width[0:-1]
#             if self.maxLane - self.maxLane > 60:
#                 width[0] = self.maxLane - self.minLane
#         self.width_road = np.average(width)
#         print('Width road: ', self.width_road)
#         self.center = int((self.minLane + self.maxLane) / 2)
#         if not self.minHead == self.maxHead == 91:
#             if self.maxLane >= self.OV_MAX_1 and self.UN_MIN_1 <= self.minLane <= self.OV_MIN_1:
#                 self.center = self.minLane + int(self.width_road / 2)
#             elif self.minLane < self.UN_MIN_1 and self.UN_MAX_1 <= self.maxLane <= self.OV_MAX_1:
#                 self.center = self.maxLane - int(self.width_road / 2)
#         self.speed = self.__PWM()
#         if float(self.speed) < 20.0:
#             self.speed = 30
#         elif float(self.speed) > optionSpeed:  # Adjust Speed
#             self.speed = underSendBack
#         if self.minLane == 450 and self.maxLane == 150 or self.maxLane == 550 or self.minLane == 0:
#             self.count += 1
#         return self.speed, self.center
#
#     def __turnRight(self):
#         if not self.corner and self.count:
#             self.corner = 1
#
#         if self.corner:
#             if time.time() - pre_time < 1.0:
#                 self.center = self.centerRight
#             else:
#                 self.trafficSigns = 'straight'
#                 self.__reset()
#
#         return self.trafficSigns, self.center
#
#     def __turnLeft(self):
#         if not self.corner and self.count:
#             self.corner = 1
#
#         if self.corner:
#             if time.time() - pre_time < 1.0:
#                 self.center = self.centerLeft
#             else:
#                 self.trafficSigns = 'straight'
#                 self.__reset()
#
#         return self.trafficSigns, self.center
#
#     def __PWM(self):
#         return -3 * abs(self.error) + 20
#
#     def __maxSpeedFunction(self):
#         return -0.125 * abs(self.error) + 20
#
#     def __controlTurning(self):
#         pass
#
#     def __reset(self):
#         self.corner = 0
#         self.check = 0
#         self.count = 0
#
#     def __call__(self, *args, **kwargs):
#         # self.MAX_SPEED = self.__maxSpeedFunction()
#         # if self.trafficSigns == 'decrease':
#         #     self.speed, self.center = self.__straight(-10, 20)
#         #     self.__reset()
#         # elif self.trafficSigns == 'straight':
#         #     self.speed, self.center = self.__straight(10, self.MAX_SPEED)
#         #     self.__reset()
#         # elif self.trafficSigns == 'no_straight':
#         #     self.speed, self.center = self.__straight(0, 10)
#         #     if not self.check:
#         #         if self.minLane <= 10:
#         #             self.check = 1
#         #         elif self.maxLane >= 600:
#         #             self.check = 2
#         #     elif self.check == 2:
#         #         self.trafficSigns, self.center = self.__turnRight()
#         #     else:
#         #         self.trafficSigns, self.center = self.__turnLeft()
#         # elif self.trafficSigns == 'turn_right' or self.trafficSigns == 'no_turn_left':
#         #     self.speed, self.center = self.__straight(0, 10)
#         #     if self.maxLane >= 134 and not self.check:
#         #         self.check = 1
#         #     elif self.check:
#         #         self.trafficSigns, self.center = self.__turnRight()
#         # elif self.trafficSigns == 'turn_left' or self.trafficSigns == 'no_turn_right':
#         #     self.trafficSigns, self.center = self.__straight(0, 10)
#         #     if self.minLane <= 25 and not self.check:
#         #         self.check = 1
#         #     elif self.check:
#         #         self.trafficSigns, self.center = self.__turnLeft()
#         # elif self.trafficSigns == 'car_right':
#         #     self.trafficSigns, self.center = self.__straight(10, self.MAX_SPEED)
#         #     self.center -= 5
#         # elif self.trafficSigns == 'car_left':
#         #     self.trafficSigns, self.center = self.__straight(10, self.MAX_SPEED)
#         #     self.center += 5
#         # if self.trafficSigns == 'straight':
#         #     print('True')
#         #     self.angle, self.speed = Controller.__call__(self)
#         # elif self.trafficSigns == 'turn_right':
#         #     print('Haha')
#         #     self.center = 600
#         #     self.error = self.computeError(self.center)
#         #     self.angle, self.speed = Controller.__call__(self)
#         # elif self.trafficSigns == 'turn_left':
#         #     print('Haha')
#         #     self.center = 0
#         #     self.error = self.computeError(self.center)
#         #     self.angle = self.PID(self.error, scale=40)
#         #     self.angle, self.speed = Controller.__call__(self)
#         # self.error = self.computeError(self.center)
#         # angle = self.PID(self.error)
#         return self.trafficSigns, self.speed, self.angle
